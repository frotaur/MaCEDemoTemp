from ..automaton import Automaton
import torch, pygame
import torch.nn.functional as F
import colorsys
from pyca.interface import MultiToggle, LabeledSlider, Button

class ConservCA(Automaton):
    """
        2D CA with conservative evolution rules
    """

    def __init__(self, size, affinity=None, device='cpu'):
        """
            Args :
            size : tuple, size of the automaton
            affinity : function, function that computes the affinity of a cell to its neighborhood
            device : str, device to use for computation
        """
        super().__init__(size)
        self.device = device

        self.c = 1

        self.kernel_size = 3 # Or make this a parameter of __init__
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        self.kernel = torch.ones(self.kernel_size, self.kernel_size, device=device, dtype=torch.float)
        self.kernel[self.kernel_size // 2, self.kernel_size // 2] = 0
        self.kernel = self.kernel[None, None, :, :].expand(self.c, -1, -1, -1)  # (c,1,kernel_size,kernel_size)
        
        if(affinity is None) :
            self.affinity = self._default_affinity
        
        self.world = torch.zeros((self.c,self.h,self.w), dtype=torch.int, device=device)
        
        self.hgrid, self.wgrid = torch.meshgrid(torch.arange(self.h,device=device),torch.arange(self.w,device=device)) # (h,w)

        self.faucet=False
        self.tapestry=False
        self.change_highlight_color()
        self.decay_speed = .1

        self.mu = 30.
        self.s = 20.

        self.echo = True
        self.prev = self.world.clone()

        self.faucet_dict={'none':'no mass addition', 'faucet':'mass faucet', 'tapestry':'mass tapestry'}
        self.inv_faucet_dict={v:k for k,v in self.faucet_dict.items()}

        self.faucet_toggle = MultiToggle(['no mass addition', 'mass faucet', 'mass tapestry'])
        self.register_component(self.faucet_toggle)
        self.reset_circle = Button('Reset to circle')
        self.register_component(self.reset_circle)
        self.reset_noise = Button('Reset to noise')
        self.register_component(self.reset_noise)
        self.wipe_button = Button('Wipe state')
        self.register_component(self.wipe_button)
        self.sigma_slider = LabeledSlider(min_value=2.,max_value=60.,title='Sigma parameter',precision=2,initial_value=self.s)
        self.register_component(self.sigma_slider)
        self.mu_slider = LabeledSlider(min_value=0., max_value=60., title='Mu parameter', precision=2, initial_value=self.mu)
        self.register_component(self.mu_slider)

        self.change_color = Button('Change color')
        self.register_component(self.change_color)
        
    def _count(self) :
        """
            Computes neighborhood count
        """
        pad_world = F.pad(self.world[None].float(), (self.kernel_size//2,self.kernel_size//2,self.kernel_size//2,self.kernel_size//2), mode='circular') # (1,c,h+2,w+2)

        return F.conv2d(pad_world, self.kernel,groups=self.c)[0].to(torch.int) # (c,h,w)

    def change_highlight_color(self):
        """
        Changes the highlight color, gets random hue
        """
        hue = torch.rand(1).item()
        light = 0.5
        saturation = 0.5
        self.highlight_color = torch.tensor(
            colorsys.hls_to_rgb(hue, saturation, light), dtype=torch.float, device='cpu'
        )

    def _default_affinity(self, count, world) :
        """
            Default affinity function. Count is outer-holistic. World is provided for outer-holistic computations
        """
        full_sum = count + world
        # Score peaks at full_sum=10, decays with distance, and becomes negative for values >50
        # Combine exponential decay with linear penalty for values far from 10
        distance = torch.abs(full_sum.float() - self.mu)
        base_score = torch.exp(-distance**2/self.s)

        return base_score.float()
        # return -torch.sigmoid(count.float()-5.)

    def _other_affinity(self, count, world) :
        full_sum = count + world
        # Score peaks at full_sum=10, decays with distance, and becomes negative for values >50
        # Combine exponential decay with linear penalty for values far from 10
        distance = torch.abs(full_sum.float() - self.mu)
        distance2 = torch.abs(full_sum.float() - self.mu/2)
        base_score = torch.exp(-distance**2/self.s) + torch.exp(-distance2**2/0.5*self.s)

        return base_score.float()

    def draw(self):
        """
            Draws the automaton
        """
        clip_value = 10
        
        cur_map = torch.clamp(self.world, 0, clip_value).float()/clip_value # (c,h,w)
        cur_map = cur_map.to('cpu')
        # if(self.c==1):
        #     self._worldmap=self._worldmap.expand(3,-1,-1)

        if(self.echo):
            echo = torch.clamp(self._worldmap - self.decay_speed * self.highlight_color[:, None, None], min=0, max=1)

            self._worldmap = torch.clamp(
                cur_map.expand(3, -1, -1).to(dtype=torch.float) + echo, min=0, max=1
            ).to('cpu')
        else:
            # display also the last step
            red_tens = torch.tensor([1.,0.,0.], device='cpu').view(3,1,1)
            blue_tens = torch.tensor([0.,1.,0.], device='cpu').view(3,1,1)
            self._worldmap = cur_map.expand(3, -1, -1).to(dtype=torch.float)*blue_tens+self.prev*red_tens# (3,h,w)
        
        self.prev = cur_map
        

    def step(self) :
        """
            Performs a step of the automaton
        """

        count = self._count()
        aff = self.affinity(count, self.world) # (c,h,w)

        distri_mask = F.pad(aff, (1,1,1,1), mode='circular') # (c,h+2,w+2)
        distri_mask = F.unfold(distri_mask[None], kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w)

        maxes = torch.max(distri_mask, dim=2,keepdim=False).values # (1,c,h,w), max of each neighborhood

        value_normalization = (distri_mask == maxes[:,:,None]).int().sum(dim=2,keepdim=False) # (1,c,h,w), number of maxes in each neighborhood

        maxes = F.pad(maxes, (1,1,1,1), mode='circular') # (1,c,h+2,w+2)
        maxes = F.unfold(maxes, kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w) # (1,c,9,h,w)

        winner_mask = (aff[None,:,None] == maxes).int() # (1,c,9,h,w) For a cell, the 3*3 tensor contains 1 if it was the max of the neighborhood

        world_share = self.world[None]//value_normalization # (1,c,h,w)
        world_remain = self.world[None]%value_normalization # (1,c,h,w)

        world_share = F.pad(world_share, (1,1,1,1), mode='circular') # (1,c,h+2,w+2)
        world_share = F.unfold(world_share.float(), kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w)

        new_world = (winner_mask * world_share.int()).sum(dim=2) # (1,c,h,w)
        new_world += world_remain

        self.world = new_world[0].int()        
    
        if self.faucet=='faucet':
            self.world[:,self.h//2-1:self.h//2+1,self.w//2-1:self.w//2+1] += torch.randint_like(self.world[:,self.h//2-1:self.h//2+1,self.w//2-1:self.w//2+1],0,2)
        elif self.faucet=='tapestry':
            self.world[:,self.h//2,self.w//2] += 2


    def reset(self,circle=False):
        """
            Resets the automaton
        """
        if(circle):
            self.circle_mask = torch.where((self.hgrid-self.h//2)**2 + (self.wgrid-self.w//2)**2 < 400, torch.randint_like(self.world,0,5), 0).to(self.device)
            self.world = self.circle_mask
        else:
            self.world = torch.randint_like(self.world,0,2).to(self.device)
    
    def wipe(self):
        """
            Wipes the automaton
        """
        self.world = torch.zeros_like(self.world).to(self.device)

    def _set_faucet(self, value):
        print('SETTING FAUCET TO : ', value)
        self.faucet = value
        self.faucet_toggle.value = self.faucet_dict[self.faucet]


    def process_event(self, event, camera=None):
        """Handles user input events.

        Keyboard:
            - BACKSPACE/DELETE: Wipe automaton.
            - O: Reset with circle.
            - I: Reset with noise.
            - T: Set to tapestry mode.
            - F: Set to faucet mode.
            - N: Set to no mass faucet.
            - Z: Change highlight color.
            - UP: Decrease decay speed.
            - DOWN: Increase decay speed.
            - E: Toggle echo mode.
        Mouse:
            - Left Click: Add matter
            - Right Click: Subtract matter
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.wipe()
            if event.key == pygame.K_o :
                self.reset(circle=True)
            if event.key == pygame.K_i :
                self.reset(circle=False)
            if event.key == pygame.K_t :
                self._set_faucet('tapestry')
            if event.key ==pygame.K_f :
                self._set_faucet('faucet')
            if event.key == pygame.K_n :
                self._set_faucet('none')
            if event.key == pygame.K_z:
                self.change_highlight_color()
            if event.key == pygame.K_UP:
                self.decay_speed = max(self.decay_speed - 0.1 * self.decay_speed, 0.005)
            if event.key == pygame.K_DOWN:
                self.decay_speed = min(0.1 * self.decay_speed + self.decay_speed, 3)
            if event.key == pygame.K_e:
                self.echo = not self.echo
        
        mouse_state = self.get_mouse_state(camera)
        if((mouse_state.left or mouse_state.right) and mouse_state.inside):
            add_rad = 10/2.
            x, y = mouse_state.x, mouse_state.y
            add_mask = (self.X-x)**2 + (self.Y-y)**2 < add_rad**2  # (H,W)

            if(mouse_state.left):
                addition = torch.randint(0, 2, (self.c,self.h, self.w), device=self.device)
                self.world[:,add_mask] += addition[:,add_mask]
            elif(mouse_state.right):
                self.world[:,add_mask] -= 1
                self.world[:,add_mask] = torch.clamp(self.world[:,add_mask], 0)
        
        for component in self.changed_components:
            if(component == self.faucet_toggle):
                val = self.inv_faucet_dict[self.faucet_toggle.value]
                self._set_faucet(val)
            if(component == self.reset_circle):
                self.reset(circle=True)
            if(component == self.reset_noise):
                self.reset(circle=False)
            if(component == self.wipe_button):
                self.wipe()
            if(component == self.mu_slider):
                self.mu = self.mu_slider.value
            if(component == self.sigma_slider):
                self.s = self.sigma_slider.value
            if(component == self.change_color):
                self.change_highlight_color()