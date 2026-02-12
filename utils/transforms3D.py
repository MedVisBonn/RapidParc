import torch
import torch.nn as nn
from typing import List
import math


class Rotate3DBatch_UnifDist(nn.Module):
    """
    Rotate the input 3D tensor around the x-, y- and z-axis. This class is designed to be used with the GPU with batch processing.
    """

    def __init__(self, range_x: List[float], range_y: List[float], range_z: List[float]):
        """
        Parameters:
            range_x (List[float]): Range of the uniform distribution for the rotation around the x-axis in degrees
            range_y (List[float]): Range of the uniform distribution for the rotation around the y-axis in degrees
            range_z (List[float]): Range of the uniform distribution for the rotation around the z-axis in degrees
        """
        super(Rotate3DBatch_UnifDist, self).__init__()
        assert len(range_x) == 2, "range_x must have length 2"
        assert len(range_y) == 2, "range_y must have length 2"
        assert len(range_z) == 2, "range_z must have length 2"
        self.range_x = torch.tensor([range_x[0], range_x[1]]) * math.pi / 180
        self.range_y = torch.tensor([range_y[0], range_y[1]]) * math.pi / 180
        self.range_z = torch.tensor([range_z[0], range_z[1]]) * math.pi / 180


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3], on device
        """
        bs = x.size(0)
        device = x.device
        thetaX = torch.rand(size=(bs,), device=device) * (self.range_x[1] - self.range_x[0]) + self.range_x[0] # Shape [bs]
        thetaY = torch.rand(size=(bs,), device=device) * (self.range_y[1] - self.range_y[0]) + self.range_y[0] # Shape [bs]
        thetaZ = torch.rand(size=(bs,), device=device) * (self.range_z[1] - self.range_z[0]) + self.range_z[0] # Shape [bs]

        rotationMatrix = \
            torch.stack([
                torch.stack([
                    torch.cos(thetaX) * torch.cos(thetaY), 
                    torch.cos(thetaX) * torch.sin(thetaY) * torch.sin(thetaZ) - torch.sin(thetaX) * torch.cos(thetaZ), 
                    torch.cos(thetaX) * torch.sin(thetaY) * torch.cos(thetaZ) + torch.sin(thetaX) * torch.sin(thetaZ)], dim=1), # Shape [bs, 3]
                torch.stack([
                    torch.sin(thetaX) * torch.cos(thetaY), 
                    torch.sin(thetaX) * torch.sin(thetaY) * torch.sin(thetaZ) + torch.cos(thetaX) * torch.cos(thetaZ), 
                    torch.sin(thetaX) * torch.sin(thetaY) * torch.cos(thetaZ) - torch.cos(thetaX) * torch.sin(thetaZ)], dim=1),
                torch.stack([
                    -torch.sin(thetaY), 
                    torch.cos(thetaY) * torch.sin(thetaZ), 
                    torch.cos(thetaY) * torch.cos(thetaZ)], dim=1)], 
            dim=2) # Shape [bs, 3, 3]
        del thetaX, thetaY, thetaZ

        return torch.einsum("bij, bspj -> bspi", rotationMatrix, x)
    

    def __repr__(self):
        return f"{self.__class__.__name__}: range_x={self.range_x}, range_y={self.range_y}, range_z={self.range_z}"



class RandomNoiseBatch(nn.Module):
    """
    Add random noise to the input tensor. This class is designed
    to be used with the GPU with batch processing.
    """
    def __init__(self, mu: float, sigma: float):
        """ 
        Parameters:
            mu (float): Mean of the normal distribution
            sigma (float): Standard deviation of the normal distribution
        """
        super(RandomNoiseBatch, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3], on device
        """
        device = x.device
        noise = torch.normal(mean=self.mu, std=self.sigma, size=x.size(), device=device)
        return x + noise
    
    def __repr__(self):
        return f"{self.__class__.__name__}: mu={self.mu}, sigma={self.sigma}"
    


class RandomStretchBatch(nn.Module):
    """
    Stretch the input 3D tensor along the x-, y- and z-axis. All Streamlines of one subject are stretched in the same way.
    This class is designed to be used on GPU with batch processing.
    """
    def __init__(self, sigma_x: float, sigma_y: float, sigma_z: float):
        """ 
        Parameters:
            sigma_x (float): Standard deviation of the normal distribution along x-axis
            sigma_y (float): Standard deviation of the normal distribution along y-axis
            sigma_z (float): Standard deviation of the normal distribution along z-axis
        
        Output:
            torch.tensor: Stretched tensor with shape [bs, numStreamlines, numPoints, 3]
        """
        super(RandomStretchBatch, self).__init__()
        self.sigmas = torch.tensor([sigma_x, sigma_y, sigma_z], requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3], on device
        """
        device = x.device
        bs = x.size(0)
        stretch = torch.normal(mean=1.0, std=self.sigmas.repeat(bs, 1)).to(device) # [bs, 3]
        return torch.einsum('bjkd,bd->bjkd', x, stretch)
    
    def __repr__(self):
        return f"{self.__class__.__name__}: sigma_x={self.sigmas[0]}, sigma_y={self.sigmas[1]}, sigma_z={self.sigmas[2]}"



class RandomFlipBatch(nn.Module):
    """
    Flip the input tensor along the Axis of the streamline supporting points. 
    This class is designed to be used with the GPU with batch processing.
    """
    def __init__(self, flipProbability: float = 0.5):
        super(RandomFlipBatch, self).__init__()
        self.flipProbability = flipProbability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3], on device
        """
        device = x.device
        bs, numStreamlines, numPoints, dim = x.size()
       
        # Decide for each streamline of a batch if it should be flipped
        isFlipPosition = torch.bernoulli(self.flipProbability * torch.ones(size=(bs, numStreamlines), device=device)) # Shape [bs, numStreamlines]
        
        # Flip the streamlines if the flipPosition is 1
        return torch.einsum("bspd, bs -> bspd", torch.flip(x, dims=[2]), isFlipPosition) + torch.einsum("bspd, bs -> bspd", x, 1 - isFlipPosition)
    
    def __repr__(self):
        return f"{self.__class__.__name__}"



def normalize_to_identity_cube(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input streamlines. This function is designed to be used with the GPU.
    The normalized streamlines should have a minimal value of -1 and a maximal value of 1 in all axis.

    Parameters:
        x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3]

    Returns:
        torch.tensor: Normalized Streamlines with shape [bs, numStreamlines, numPoints, 3]
    """
    bs, _, _, dim = x.size()
    mins = torch.amin(x, dim=(1,2)).view(bs, 1, 1, dim)
    maxs = torch.amax(x, dim=(1,2)).view(bs, 1, 1, dim)

    return  2 * (x - mins) / (maxs - mins) -1.0



class RandomNoisedNormalizeToIdentetyCube(nn.Module):
    """
    Normalize the input streamlines. But it applies some random noise. This class is designed to be used with the GPU.
    The normalized streamlines should have a minimal value of -1 and a maximal value of 1 in all axis.
    """
    def __init__(self, shift_std: float = 0.01, scale_std: float = 0.01):
        """ 
        Parameters:
            shift_std (float): Standard deviation of the normal distribution for the shift
            scale_std (float): Standard deviation of the normal distribution for the scale
        """
        super(RandomNoisedNormalizeToIdentetyCube, self).__init__()
        self.shift_std = shift_std
        self.scale_std = scale_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.tensor): Shape [bs, numStreamlines, numPoints, 3], on device
        """
        bs, _, _, dim = x.size()
        mins = torch.amin(x, dim=(1,2)).view(bs, 1, 1, dim)
        maxs = torch.amax(x, dim=(1,2)).view(bs, 1, 1, dim)

        shift_noise = torch.normal(mean=-1.0, std=self.shift_std, size=(bs, dim), device=x.device).view(bs, 1, 1, dim)
        scale_noise = torch.normal(mean=2.0, std=self.scale_std, size=(bs, dim), device=x.device).view(bs, 1, 1, dim)

        return  scale_noise * (x - mins) / (maxs - mins) + shift_noise
    
    def __repr__(self):
        return f"{self.__class__.__name__}: shift_std={self.shift_std}, scale_std={self.scale_std}"