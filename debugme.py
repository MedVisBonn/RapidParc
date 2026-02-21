from rapidparc import rapidParc, test
import torch
from rapidparc.utils.pypi_package_helper import get_tractCloud_dataset_path

# inputTractogram = torch.rand([5000, 15, 3])

# solution = rapidParc(model_name_or_path="rapidparc",
#           inputTractogram=inputTractogram,
#           eval_batch_size=4,
#           eval_context_size=1000,
#           device=torch.device("cpu")
#           )

# print(solution)


# filePath = get_tractCloud_dataset_path()

# print(filePath)

test(model_name_or_path="rapidparc",
     applyTestSetAugmentations=True)