import torch, torch.nn.functional as F
from ..utils import torch_utils
from ..automaton import Automaton
import cv2
from pathlib import Path
import pygame
from pyca.interface.ui_components import Button, MultiToggle, Toggle

class MaCENCA(Automaton):
    """
        NCA using MaCE in its update rule. RGB channels conserve mass.
    """
    def __init__(self, size, saved_folder=None, device='cpu'):
        """
            Args:
            size : (H,W) size of the NCA world
            saved_folder : str, path to the folder containing the saved weights and target image
            device : str, device to run the model on
        """
        super().__init__(size)
        self.device = device
        self.saved_weights = [] # Tuples (weights_path, image_path)
        for file in Path(saved_folder).glob("*.pth"):
            associated_image = file.with_suffix('.png')
            if associated_image.exists():
                print('Added : ', str(file), str(associated_image))
                self.saved_weights.append((str(file), str(associated_image)))

        self.loaded_model = 0

        if(len(self.saved_weights) > 0):
            self.load_model(self.loaded_model)
        else:
            C = 16
            self.model = MassConservingNCA(C,hidden_n=128, device=device).to(device)
            self.seed = torch.zeros((C), device=device)
            self.seed[:3] = 20.

        self.state = torch.zeros(1, self.model.C, size[0], size[1], device=device)

        self.put_seed_at(size[1]//2, size[0]//2)

        self.add_speed = 1.0

        self.wipe_button = Button(text="Wipe")
        self.register_component(self.wipe_button)
        self.next_model = Button(text="Next Model")
        self.register_component(self.next_model)
        self.seed_next = Toggle(state1="Click anywhere to plant seed", state2="Plant seed", init_true=False)
        self.register_component(self.seed_next)
        self.add_matter = Toggle(state1="Adding mass", state2="Removing mass", init_true=True)
        self.register_component(self.add_matter)

    def put_seed_at(self, x,y):
        self.state[0,:,y,x] = self.seed # (1,C,H,W)
        
    @torch.no_grad()
    def step(self):
        self.state = self.model(self.state)
    
    def load_model(self, index):
        index = index % len(self.saved_weights)
        weight_path, img_path = self.saved_weights[index]
        print('LOADING : ', weight_path, img_path)
        weights = torch.load(weight_path, map_location=self.device)
        params = self.get_params(weights)
        self.model = MassConservingNCA(params['C'],hidden_n=params['hidden_n'], device=self.device).to(self.device)
        self.model.load_state_dict(weights, strict=True)
        self.seed = self.get_seed(img_path).to(self.device) # (C,1,1)
    
    def draw(self):
        self._worldmap = torch.clamp(self.state[0,:3],min=0, max=1) # (3,H,W)
        temp = self._worldmap[0].clone()
        self._worldmap[0] = self._worldmap[2]
        self._worldmap[2] = temp

    def get_params(self,weights):
        return {'C': weights['w2.weight'].shape[0], 'hidden_n': weights['w1.weight'].shape[0]}
    
    def get_seed(self,img_path, tgt_size=(50,50)):
        """
            Given an image path, computes the seed (size (C,)) with all channels
            set to zero, except RGB which contain all necessary mass.
        """
        base = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        base = cv2.resize(base, (int(tgt_size[0]), int(tgt_size[1])), interpolation=cv2.INTER_LINEAR)

        base_2 = base / 255
        
        base_2[..., :3] *= base_2[..., 3:]
        base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).to(self.device)
        
        seed = torch.zeros((self.model.C), device= self.device)
        seed[:3] = base_torch[:3].sum()/3 # Set the mass in RGB channels, average value

        return seed

    def process_event(self, event, camera=None):
        """
        Processes a keyboard event.
        'r' : reset the automaton
        's' : step the automaton
        'd' : toggle dot mode
        'c' : change highlight color
        """
        mouse = self.get_mouse_state(camera)

        if mouse.right and mouse.inside:
            self.add_or_remove_mass(mouse.x, mouse.y, add=False)

        
        if mouse.left and mouse.inside:
            if(self.seed_next.state): 
                self.put_seed_at(mouse.x, mouse.y)
                self.seed_next.state = False
            elif(self.add_matter.state):
                self.add_or_remove_mass(mouse.x, mouse.y, add=True)
            elif(self.add_matter.state == False):
                self.add_or_remove_mass(mouse.x, mouse.y, add=False)
    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                self.loaded_model = (self.loaded_model + 1) % len(self.saved_weights)
                self.load_model(self.loaded_model)
            if event.key == pygame.K_p:
                self.seed_next.state = True
            if event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                self.wipe()

        for component in self.changed_components:
            if component == self.wipe_button:
                self.wipe()
            if component == self.next_model:
                self.loaded_model = (self.loaded_model + 1) % len(self.saved_weights)
                self.load_model(self.loaded_model)
                
    
    def add_or_remove_mass(self, x, y, add=True):
        """
            Adds or removes mass at position (x,y) in the world.
        """
        add_rad = 10.
        add_mask = (self.X - x) ** 2 + (self.Y - y) ** 2 < add_rad**2  # (H,W)

        if add:
            addition = torch.rand((1, self.model.C, self.h, self.w), device=self.device)
            self.state[:, :, add_mask] += self.add_speed *0.05 * addition[:, :, add_mask]
        else:
            self.state[:, :, add_mask] -= 0.05*self.add_speed
            self.state[:, :, add_mask] = torch.clamp(self.state[:, :, add_mask], 0.0, 1.0)

    
    def wipe(self):
        self.state = torch.zeros(1, self.model.C, self.h, self.w, device=self.device)

class MassConservingNCA(torch.nn.Module):
    def __init__(self, C,hidden_n, device):

        torch.nn.Module.__init__(self)

        super(MassConservingNCA, self).__init__()
        self.C = C
        self.w1 = torch.nn.Conv2d(4 * C, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, C, 1, bias=False)
        self.w2.weight.data.zero_()
        self.device = device


        self.ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        self.ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        self.sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        self.lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device=device)


    @torch.no_grad()
    def forward(self, x, update_rate=1):
        y = self.perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()

        x_normal = x[:,3:,...]
        x_mass = x[:,:3,...]

        y_normal = y[:,3:,...]
        y_mass = y[:,:3,...]

        Aff = torch.exp(y_mass*0.1)

        x_mass = self.redistribution(Aff, x_mass)
        x_normal = x_normal + y_normal * update_mask

        x = torch.cat((x_mass, x_normal), dim = 1)

        return x

    def redistribution(self,Aff,state):

        B, C, H, W = state.shape
        Aff_exp = F.pad(Aff, (1, 1, 1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        Aff_exp = torch_utils.unfold3d(Aff_exp, kernel_size=(3, 3, 3)).reshape(B, C, 27, H, W)  # (B,C*9,H,W)
        E = Aff_exp.sum(dim=2)
        E_exp = F.pad(E, (1, 1, 1, 1, 1, 1), mode="circular")
        E_exp = torch_utils.unfold3d(E_exp, kernel_size=(3, 3, 3)).reshape(B, C, 27, H, W)  # (B,C*9,H,W)
        state_exp = F.pad(state, (1, 1, 1, 1, 1, 1), mode="circular")
        state_exp = torch_utils.unfold3d(state_exp, kernel_size=(3, 3, 3)).reshape(B, C, 27, H, W)  # (B,C*9,H,W)

        state = ((Aff[:, :, None, ...] / E_exp) * state_exp).sum(dim=2)

        return state

    def perception(self,x):
        filters = torch.stack([self.sobel_x, self.sobel_x.T, self.lap])
        obs = self.perchannel_conv(x, filters)
        return torch.cat((x,obs), dim = 1 )
    
    def perchannel_conv(self, x, filters):
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

