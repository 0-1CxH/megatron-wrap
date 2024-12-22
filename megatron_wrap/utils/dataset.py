import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class BatchDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.start_index = 0
    
    def get_batch(self, batch_size):
        assert self.__len__() >= batch_size
        if self.start_index + batch_size > self.__len__():
            self.start_index = 0
        ret = [self[idx] for idx in range(self.start_index, self.start_index+batch_size)]
        self.start_index += batch_size
        return ret


class EmptyDataset(BatchDataset):
    def __len__(self):
        return 2 ** 16
    
    def __getitem__(self, index):
        return [index]
    

class ExampleSftDataset(BatchDataset):
    def __init__(
            self, 
            example_jsonl_path="sample/tldr_3200.jsonl",
            tokenizer_path="sample/llama2_32k_tokenizer_files"
        ):
        super().__init__()
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        with open(example_jsonl_path) as f:
            for line in f.readlines():
                self.data.append(
                    json.loads(line)
                )
    
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, index):
        raw_data = self.data[index]
        input_ids = self.tokenizer.apply_chat_template(raw_data["messages"])
        response_length = len(self.tokenizer.encode(raw_data["messages"][-1]["content"])) + 2
        return {
            "input_ids": input_ids,
            "context_token_length": len(input_ids) - response_length
        }
        

        


