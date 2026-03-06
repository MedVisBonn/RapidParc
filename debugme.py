from rapidparc import rapidParc, test, train
import torch
from rapidparc.utils.pypi_package_helper import get_tractCloud_dataset_path

inputTractogram = torch.rand([5000, 15, 3])

print(rapidParc(model_name_or_path="rapidparc",
          inputTractogram=inputTractogram,
          eval_batch_size=4,
          eval_context_size=1000,
          device=torch.device("cpu")
          ))


# print(test(model_name_or_path="rapidparc",
#      applyTestSetAugmentations=False,
#      eval_batch_size=64,
#      device=torch.device("mps")))


# t = torch.rand(15, 3, 4)
# t2 = t.repeat_interleave(5, dim=0)
# print(t2)
