## CHGAT (Chinese Character Graph Attention Network) 

This is the source code for paper 'For the Underrepresented in Gender Bias Research: Chinese Name Gender Prediction with Heterogeneous Graph Attention Network'.

## Abstract

Achieving gender equality is an important pillar for humankind’s sustainable future. Pioneering data-driven gender bias research is based on large-scale public records such as scientific papers, patents, and company registrations, covering female researchers, inventors and entrepreneurs, and so on. Since gender information is often missing in relevant datasets, studies rely on tools to infer genders from names. However, available open-sourced Chinese gender-guessing tools are not yet suitable for scientific purposes, which may be partially responsible for female Chinese being underrepresented in mainstream gender bias research and affect their universality. Specifically, these tools focus on character-level information while overlooking the fact that the combinations of Chinese characters in multi-character names, as well as the components and pronunciations of characters, convey important messages. As a first effort, we design a Chinese Heterogeneous Graph Attention (CHGAT) model to capture the heterogeneity in component relationships and incorporate the pronunciations of characters. Our model largely surpasses current tools and also outperforms the state-of-the-art algorithm. Last but not least, the most popular Chinese name-gender dataset is single-character based with far less female coverage from an unreliable source, naturally hindering relevant studies. We open-source a more balanced multi-character dataset from an official source together with our code, hoping to help future research promoting gender equality.

## Data

We provide a dataset with 58,393,173 records of 560,706 different first names and the associated gender for each name occurrence,  collected from an official source. You can download the data from the folder `58M Names`.

## Prerequisites

The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)

 - Python 3.8.1
 - PyTorch 1.11.0
 - dgl 0.9.0
 - Sklearn 1.1.2
 - Pandas 1.4.3
 - Transformers 4.21.1
 - Tensorboard 2.10.0

## Getting Started

### Prepare your data

1. The example of `word2pinyin2real_word2formation2sem2pho_dict.pkl`. This file is generated from [汉字全息资源应用系统](https://qxk.bnu.edu.cn/#/).
```python
{'pinyin': ['zhu1'],
 'real_word': '珠（73E0）',
 'label_formation': [['珠（73E0）', 0], ['王（738B）', 1], ['朱（6731）', 2]],
 'sem_component': ['王（738B）', 1],
 'pho_component': ['朱（6731）', 2]}
```
2. The example of `word_and_component_vocab_list.pkl`. The file records characters from [汉字全息资源应用系统](https://qxk.bnu.edu.cn/#/).
```python
['倔（5014）',
 '學（5B78）',
 '焫（712B）',
 '廢（5EE2）',
 '钍（948D）',
 '瞞（779E）',
 '缸（7F38）',
  ...]
```

3. The example of `pinyin_index_dict.pkl`. The file records the index of the pronunciation.
```python
{'zhu1': 0,
 'ling1': 1,
 'shun3': 2,
 'cha3': 3,
 'chuan1': 4,
 'kun3': 5,
 'dao2': 6,
 'yu3': 7,
 'ping2': 8,
 ...
}
```

4. The example of `chgat_formation_graph.pkl`. This file is the required data of the construction of the Chinese character graph.
```python
{'node_list': [['珠（73E0）', 0, 0],
  ['王（738B）', 1, 1],
  ['朱（6731）', 2, 2],
  ['琇（7407）', 0, 3],
  ['瓔（74D4）', 0, 4],
  ['瓔（74D4）', 0, 5],
  ['瑰（7470）', 0, 6],
  ['瑪（746A）', 0, 7],
  ['姝（59DD）', 0, 8],
  ['硃（7843）', 0, 9],
  ['銖（9296）', 0, 10],
  ['銖（9296）', 0, 11],
  ['株（682A）', 0, 12],
  ['zhu1', 13]],
 'sem_list': [[[0, 1, 1, 1, 1, 1], [1, 3, 4, 5, 6, 7]]],
 'pho_list': [[[0, 2, 2, 2, 2, 2], [2, 8, 9, 10, 11, 12]]],
 'non_sp_list': [[[], []]],
 'pinyin_list': [[[0], [13]]],
 'graph': Graph(num_nodes={'c': 14},
       num_edges={('c', 'c-pho', 'c'): 6, ('c', 'c-pinyin', 'c'): 1, ('c', 'c-sem', 'c'): 6},
       metagraph=[('c', 'c', 'c-pho'), ('c', 'c', 'c-pinyin'), ('c', 'c', 'c-sem')]),
 'node_word_list': [['珠（73E0）', 0, 0],
  ['王（738B）', 1, 1],
  ['朱（6731）', 2, 2],
  ['琇（7407）', 0, 3],
  ...]
 'node_pinyin_list': [['zhu1', 13]]}
```

5. Download the dataset from the link [google driver](https://drive.google.com/drive/folders/1TX5dAwE6_v2AcgBx2Ngwb7kUdBFAlhC-?usp=sharing) or prepare your own dataset in a similar format. We provide the training data, validation data and test data.

### Training CHGAT
Please run following commands for training.
```python
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 chgat.py --task='chgat'
```

## Cite
Please cite our paper if you find this code useful for your research:



