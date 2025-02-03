import fire
import random
import itertools
import math
from pathlib import Path, PurePath
from typing import List, Union, Tuple, Any, Dict
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import wandb
import rasterio
import torch
from torch.backends import cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import grad_scaler


class FSToolKit:

    def __init__(self):
        pass

    @staticmethod
    def path2str(path: Path) -> str:
        return str(path.resolve())

    @staticmethod
    def read_from_file(path: Union[str, Path], parent: str = "") -> List[str]:
        content = []
        with Path(path).open("r") as f:
            content = f.readlines()
        content = [
            str(PurePath(parent).joinpath(line.strip()))
            for line in content if len(line.strip()) > 0
        ]
        return content

    @staticmethod
    def save2file(filepath: Path, content: List[str]) -> None:
        with filepath.resolve().open("w") as f:
            for line in content:
                f.write(f"{line}\n")


class EroDataset(Dataset):

    def __init__(self, data_path, split_path):
        self.split_path = split_path

        content = FSToolKit.read_from_file(self.split_path, parent=data_path)
        content = [Path(line) for line in content]  # if filtering desired, do it here
        self.folder_path_list = content

        self.outputs = self.get_outputs()

    def get_outputs(self) -> npt.NDArray[np.float32]:
        where_splits = Path(self.split_path).parent.absolute()

        ofp = where_splits.joinpath('outputs')  # stands for "outputs file path"
        ofp.mkdir(exist_ok=True)
        ofp = ofp.joinpath(Path(self.split_path).name)
        print(f"{ofp = }")

        if ofp.exists():
            # read the outputs from file, and return them
            content = FSToolKit.read_from_file(ofp)
            outputs = [float(line.rstrip()) for line in content]  # filtering here
        else:
            # create the outputs, save them to file, and return them
            outputs = []
            for path in tqdm(self.folder_path_list):
                with path.joinpath("label.txt").open('r') as file:
                    # read output value from the file (contains only this)
                    output = float(file.read().strip())
                outputs.append(output)
            FSToolKit.save2file(ofp, outputs)
        return np.array(outputs)
    
    def read_data(self, folder_path_list: List[Path]) -> torch.Tensor:
        data = []
        for path in folder_path_list:
            bands_data = []
            for suffix in ['_B04_10m.tiff', '_B03_10m.tiff', '_B02_10m.tiff']: 
                for ipath in path.iterdir():
                    if ipath.is_file() and suffix in ipath.name:  # never not end
                        with rasterio.open(ipath, driver="GTiff", sharing=False) as band_file:
                            band_data = band_file.read(1)  # open the tiff file as a numpy array
                        # no resizing needed since we all our bands have the same resolution
                        band_data = (band_data / 255.).astype(np.float32)  # normalization
                        # add to the pile of bands
                        bands_data.append(band_data)
            if len(bands_data) != 3:
                raise ValueError(f"invalid datapoint: {path}")
            bands_data = np.stack(bands_data) 
            data.append(bands_data)
        # encapsulate in numpy array
        data_np = np.array(data)  # useful step for debug sanity checks
        if len(data_np) == 1:
            return torch.Tensor(data_np[0])
        else:
            return torch.Tensor(data_np)

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, List[torch.Tensor]],
                                               torch.Tensor]:

        data = self.read_data([self.folder_path_list[index]])
        output = torch.Tensor([self.outputs[index]])
        return (data, output)


class EroDataloader(DataLoader):

    def __init__(self, dataset: EroDataset, batch_size: int, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.dataset_length = len(dataset)

    def __len__(self):
        """Overwrite because relays the method of the `Dataset` class otherwise"""
        if self.batch_size is None:
            raise ValueError(f"invalid batch size ({self.batch_size}); can't be None!")
        return math.ceil(self.dataset_length // self.batch_size)


class DLToolKit:

    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(data_path: Path, split_path: Path,
                       batch_size: int, shuffle: bool) -> EroDataloader:
        dataloader = EroDataloader(
            EroDataset(data_path, split_path),
            batch_size=batch_size,
            num_workers=0,  # a value of 0 plugs the memory leak (can't avoid using Python lists)
            shuffle=shuffle,
            drop_last=True,
        )
        return dataloader


class Model(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.global_avgpool(x)
        x = self.flatten(x)

        x = self.fc(x)
        return x


class MagicAlgo:

    def __init__(self, args: Dict[Any, Any]):
        self.args = args
        self.device = args['device']
        self.iters_so_far = 0
        self.model = Model().to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters in model: {num_params}")

        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        self.scaler = grad_scaler.GradScaler(enabled=True)  # for half-precision

    def send_to_dash(self, loss, *, step, glob):
        wandb.log({f"{glob}/loss": loss}, step)

    def compute_loss(self, x, true_y):
        pred_y = self.model(x)
        loss = self.criterion(pred_y, true_y)
        return loss

    def train(self, trai_dataloader, vali_dataloader):
        
        agg_iterable = zip(
            tqdm(trai_dataloader),
            itertools.chain.from_iterable(itertools.repeat(vali_dataloader)),
            strict=False,
        )

        for i, ((t_x, t_true_y), (v_x, v_true_y)) in enumerate(agg_iterable):

            t_x = t_x.pin_memory().to(self.device, non_blocking=True)
            t_true_y = t_true_y.pin_memory().to(self.device, non_blocking=True)

            with self.ctx:
                t_loss = self.compute_loss(t_x, t_true_y)

            self.send_to_dash(t_loss, step=self.iters_so_far, glob='trai')

            t_loss: Any = self.scaler.scale(t_loss)  # silly trick to bypass broken
            # torch.cuda.amp type hints (issue: https://github.com/pytorch/pytorch/issues/108629)
            t_loss.backward()

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if ((i + 1) % self.args['eval_every'] == 0) or (i + 1 == len(trai_dataloader)):

                self.model.eval()

                with torch.no_grad():

                    v_x = v_x.pin_memory().to(self.device, non_blocking=True)
                    v_true_y = v_true_y.pin_memory().to(self.device, non_blocking=True)

                    with self.ctx:
                        v_loss = self.compute_loss(v_x, v_true_y)

                    self.send_to_dash(v_loss, step=self.iters_so_far, glob='vali')

                self.model.train()

            self.iters_so_far += 1

    def test(self, test_dataloader):

        self.model.eval()

        with torch.no_grad():

            losses = []

            for x, true_y in test_dataloader:

                x = x.pin_memory().to(self.device, non_blocking=True)
                true_y = true_y.pin_memory().to(self.device, non_blocking=True)

                with self.ctx:
                    loss = self.compute_loss(x, true_y)
                    losses.append(loss.numpy(force=True))

            print(f"the average loss over the test set is {np.mean(losses)}")

    def save(self, ckpt: torch.Tensor, epoch: int):
        torch.save({
            'hps': self.args,
            'msd': self.model.state_dict(),
            'osd': self.opt.state_dict(),
        }, f"{str(ckpt)}_e_{epoch}")


class Orchestrator:

    def __init__(self):
        pass

    @staticmethod
    def uuid(num_syllables: int = 2, num_parts: int = 3):
        # randomly create a semi-pronounceable uuid
        part1 = ['s', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr']
        part2 = ['a', 'oo', 'ee', 'e', 'u', 'er']
        seps = ['_']  # [ '-', '_', '.']
        result = ""
        for i in range(num_parts):
            if i > 0:
                result += seps[random.randrange(len(seps))]
            indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
            indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
            for i1, i2 in zip(indices1, indices2, strict=True):
                result += part1[i1] + part2[i2]
        return result

    @staticmethod
    def wandb(args: Dict[Any, Any]):
        wandb.init(
            project=args['wandb_project'],
            name=args['experiment_name'],
            id=args['experiment_name'],
            # group=group,  # we do not use groups (e.g. seed spectrum aggregation)
            config=args,
            dir=args['root'],
        )

    @staticmethod
    def split(data: Path, sout: Path) -> List[Path]:
        ll = ['trai', 'vali', 'test'] 
        oo = [sout / f"{el}.txt" for el in ll]
        if sout.exists():
            # if split file exists already
            print(f"{sout}: already exists!")
        else:
            # make split dir
            sout.mkdir(parents=True, exist_ok=True)

            # make the list of records (folders)
            # go over these and check if they contain the tiffs we want
            invalids = []
            records = []
            for record in data.iterdir():
                c = 0
                for e in record.iterdir():
                    if e.is_file() and e.suffix == '.tiff':
                        c += 1
                if c != 3:
                    invalids.append(record)
                else:
                    # add the validated record to records to use
                    records.append(record.name)
            print(f"there are {len(invalids)} invalid records")
            print(f"and {len(records)} valid ones")

            # shuffle
            random.shuffle(records)  # shuffles the seq in place
            # calculate the sizes
            tot = len(records)
            trai_size = int(0.7 * tot)
            vali_size = int(0.1 * tot)
            test_size = tot - trai_size - vali_size  # for logging
            print(f"{trai_size=} ** {vali_size=} ** {test_size=}")
            # split the list of records
            splits = {}
            splits['trai'] = records[:trai_size]
            splits['vali'] = records[trai_size:trai_size + vali_size]
            splits['test'] = records[trai_size + vali_size:]
            # write the record lists to disk
            for el in ll:
                (sout / f"{el}.txt").write_text('\n'.join(splits[el]))
        # return the paths to the split files in either case
        return oo


    @staticmethod
    def gogo(args: Dict[Any, Any]):
        # create save dir
        ckpt = args['root'] / "checkpoints" / args['experiment_name']
        ckpt.mkdir(parents=True, exist_ok=True)
        # split the dataset
        data = args['records_dir']
        sout = args['root'] / "splits" / "70-15-15"
        splits = Orchestrator.split(data, sout)
        # create dataloaders
        kwargs = {'data_path': data, 'batch_size': args['batch_size']}
        dataloaders = [  # for train, val and test sets
            DLToolKit.get_dataloader(**kwargs, split_path=splits[0], shuffle=True),
            DLToolKit.get_dataloader(**kwargs, split_path=splits[1], shuffle=False),
            DLToolKit.get_dataloader(**kwargs, split_path=splits[2], shuffle=False),
        ]
        # sanity check
        for dl in dataloaders:
            print(len(dl))
        # create algo object (model created inside it)
        algo = MagicAlgo(args)
        # setup wand
        Orchestrator.wandb(args)
        # go over epochs
        for i in range(args['epochs']):
            # train
            algo.train(*dataloaders[0:2])
            # save weights
            algo.save(ckpt, i)
        # test the model with the algo
        algo.test(dataloaders[2])


class MagicShow:

    def __init__(self):
        pass

    def run_the_show(self, records_dir: str):
        root = Path(__file__).resolve().parent  # make path absolute
        args = defaultdict(Any)
        args['root'] = root
        args['experiment_name'] = Orchestrator().uuid()
        args['wandb_project'] = "erosion"
        args['records_dir'] = root / Path(records_dir)
        args['epochs'] = 2  # hp!?
        args['batch_size'] = 64  # hp!?
        args['lr'] = 1e-4  # hp!?
        args['eval_every'] = 20  # hp!?
        # handle device
        assert torch.cuda.is_available()
        cudnn.benchmark = False
        cudnn.deterministic = True
        args['device'] = torch.device("cuda:0")
        # seedify
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # run the magic show
        Orchestrator.gogo(args)


if __name__ == '__main__':
    fire.Fire(MagicShow)


# CLI command to run:
# python train.py run_the_show records

