import numpy as np
import torch
import torch.utils.data as data
import pickle

class EmbDataset(data.Dataset):

    def __init__(self,data_path_1,data_path_2):

        with open(data_path_1, 'rb') as f:
            self.text = pickle.load(f)
        with open(data_path_2, 'rb') as f:
            self.kg = pickle.load(f)

        # 转为 NumPy array（如果还不是）
        if not isinstance(self.text, np.ndarray):
            self.text = np.array(self.text.cpu())
        if not isinstance(self.kg, np.ndarray):
            self.kg = np.array(self.kg.cpu())
        
        print(f"Text embeddings shape: {self.text.shape}")
        print(f"KG embeddings shape: {self.kg.shape}")
        
        # self.text = torch.load(data_path_1).to('cpu').detach().numpy()
        # self.kg = torch.load(data_path_2).to('cpu').detach().numpy()

        self.text_dim = self.text.shape[-1]
        self.kg_dim = self.kg.shape[-1]

    def __getitem__(self, index):
        text = self.text[index]
        kg = self.kg[index]
        return torch.FloatTensor(text), torch.FloatTensor(kg), index

    def __len__(self):
        return len(self.text)
