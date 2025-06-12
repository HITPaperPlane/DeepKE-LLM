llm/目录是一个用llm来构建知识图谱的模块，其中llm/README.md是对这个模块的介绍

现在的场景是技术选型已经结束，
# 模型
决定基于qwen驱动知识图谱的构建。qwen已经部署完毕，可以通过api来调用，调用示例如下：

```python
from openai import OpenAI

api_base = "http://192.168.11.218:8192/v1"
model_name = "novel"

client = OpenAI(
    api_key="EMPTY",
    base_url=api_base,
)

character_intro_ = "张三，男，30岁，警察。"
chapter_text_ = "张三在街头巡逻，发现有人在偷钱包，于是他立刻上前制止。"
prompt = "角色介绍：{character_intro}\n小说内容：{text}\n请将上述小说内容改写为剧本对话。"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一位专业的戏剧编剧，擅长将小说改编为话剧剧本。"},
        {"role": "user", "content": prompt.format(character_intro=character_intro_, text=chapter_text_)}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

# 数据集
数据集采用InstrueIE和IEPile
## InstrueIE
InstrueIE的数据集结构：
```
InstrueIE
├── train_zh.json          # Chinese training set.
├── train_en.json          # English training set.
├── valid_zh.json            # Chinese validation set.
├── valid_en.json            # English validation set.
├── test_zh.json           # Chinese test set.
├── test_en.json           # English test set.
├── schema_zh.json         # Schema information for 12 domains in Chinese.
├── schema_en.json         # Schema information for 12 domains in English.
```
路径存储在dataset/InstructIE。

示例：
```json
{
  "id": "841ef2af4cfe766dd9295fb7daf321c299df0fd0cef14820dfcb421161eed4a1", 
  "text": "NGC1313 is a galaxy in the constellation of Reticulum. It was discovered by the Australian astronomer James Dunlop on September 27, 1826. It has a prominent uneven shape, and its axis does not completely revolve around its center. Near NGC1313, there is another galaxy, NGC1309.", 
  "relation": [
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "time of discovery", "tail": "September 27, 1826", "tail_type": "time"}, 
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "discoverer or inventor", "tail": "James Dunlop", "tail_type": "organization/human"}, 
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "of", "tail": "Reticulum", "tail_type": "astronomical object type"}
  ]
}

使用的更详细说明见：<https://github.com/zjunlp/EasyInstruct/blob/main/examples/kg2instruction/README.md>

```
## IEPile
IEPile的数据集结构：
```
IEPile
├── train.json      # Training Set
├── dev.json        # Validation Set
├── IE-en           # English Unified Format Data
│   ├── NER
│   │   ├── CoNLL2003
│   │   │   ├── train.json
│   │   │   ├── dev.json
│   │   │   ├── schema.json   # schema information file
│   │   │   └── test.json
│   │   ├── ...
│   ├── RE
│   ├── EE
│   ├── EET
│   ├── EEA
├── IE-zh           # Chinese Unified Format Data
│   ├── NER
│   ├── RE
│   ├── EE
│   ├── EET
│   ├── EEA
```
路径存储在dataset/iepile。

示例：
```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}

{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}
```


详细说明


[**使用方法**](https://github.com/zjunlp/IEPile) |


> 请注意，我们的IEPile可能会进行**更新**（一旦发布更新，我们将通知您）。建议使用最新版本。


- [IEPile：大规模信息提取语料库](#iepile大规模信息提取语料库)
  - [1.介绍](#1介绍)
  - [2.数据](#2数据)
    - [2.1IEPile的构造](#21iepile的构造)
    - [2.2IEPile的数据格式](#22iepile的数据格式)
  - [3.使用IEPile训练模型](#3使用iepile训练模型)
  - [4.声明和许可](#4声明和许可)
  - [5.局限](#5局限)
  - [6.引用](#6引用)
  - [7.致谢](#7致谢)
 
```
IEPile
├── train.json      # 训练集
├── dev.json        # 验证集
├── IE-en           # 英文统一格式数据
│   ├── NER
│   │   ├── CoNLL2003
│   │   │   ├── train.json
│   │   │   ├── dev.json
│   │   │   ├── schema.json   # schema信息文件
│   │   │   └── test.json
│   │   ├── ...
│   ├── RE
│   ├── EE
│   ├── EET
│   ├── EEA
├── IE-zh           # 中文统一格式数据
│   ├── NER
│   ├── RE
│   ├── EE
│   ├── EET
│   ├── EEA
```


## 🎯1.介绍


> 请注意，以上提供的数据集中所含数据已经排除了与ACE2005数据集相关的部分。若您需要访问未经过滤的完整数据集，并且已成功获取所需的权限，敬请通过电子邮件方式联系 guihonghao@zju.edu.cn 或 zhangningyu@zju.edu.cn。我们将提供完整数据集资源。


**`LLaMA2-IEPile`** | **`Baichuan2-IEPile`** | **`KnowLM-IE-v2`** 模型下载链接：[zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | [zjunlp/KnowLM-IE-v2]()


![statistic](./assets/statistic.jpg)

我们精心收集并清洗了现有的信息提取（IE）数据，共整合了`26`个英文IE数据集和`7`个中文IE数据集。如图1所示，这些数据集覆盖了包括**通用**、**医学**、**金融**等多个领域。

本研究采用了所提出的“`基于schema的轮询指令构造方法`”，成功创建了一个名为 **IEPile** 的大规模高质量IE微调数据集，包含约`0.32B` tokens。

基于**IEPile**，我们对 `Baichuan2-13B-Chat` 和 `LLaMA2-13B-Chat` 模型应用了 `Lora` 技术进行了微调。实验证明，微调后的 `Baichuan2-IEPile` 和 `LLaMA2-IEPile` 模型在全监督训练集上成绩斐然，并且在**零样本信息提取任务**中取得了提升。


![zero_en](./assets/zero_en.jpg)

![zero_zh](./assets/zero_zh.jpg)


<details>
  <summary><b>全监督数据集结果</b></summary>

![supervision_ner](./assets/supervision_ner.jpg)

![supervision_re](./assets/supervision_re.jpg)

![supervision_ee](./assets/supervision_ee.jpg)

</details>


## 📊2.数据


### 2.1IEPile的构造

我们专注于**基于指令的信息抽取**，因此指令中的schema的构造至关重要，因为它反映着具体抽取需求，是动态可变的。然而，现有研究在构造指令时往往采取一种**较为粗放的schema处理策略**，即利用标签集内**全部schema**进行指令构建。这种方法潜在地存在2个重要的问题：
1. **训练和评估阶段schema询问的数量不一致，即使这些schema在内容上相似，可能损害模型的泛化能力**。若训练过程中每次询问的schema数量大约是20个，而评估时询问的是10个或30个schema，即使这些schema在内容上与训练阶段相似，模型性能仍可能受到影响。
2. **指令中的schema之间的对比性不足**。语义近似的schema，如“裁员”、“离职”与“解雇”，它们的语义模糊性可能造成模型混淆。这类易混淆的模式应当在指令集中更为频繁地出现。
   
因此，我们提出如下解决方案：1、`构造难负样本字典`；2、`轮询式的指令生成`。

![iepile](./assets/iepile.jpg)


<details>
  <summary><b>难负样本</b></summary>


假设数据集 $\mathcal{D}$ 有其全量标签集 $L$，$\mathcal{D}$ 中某一文本 $S$，$S$ 中真实存在的标签构成**正例标签集** $Pos\_L$，而不存在的标签则形成**负例标签集** $Neg\_L$。在我们的分析中，我们发现模型误判的主要原因在于schema的**语义模糊**，导致了模型的混淆。传统方法中，负例标签 $Neg\_L$通常简单地定义为 $L - Pos\_L$。然而，这种方法忽视了一个重要方面：需要特别注意那些与正例标签**语义相近**的负例标签。受**对比学习**理论的启发。我们构造了一个**难负样本字典** $\mathcal{D}$，其键值对应的是Schema及其语义上相近的Schema集。因此**难负样本集** $Hard\_L = \mathcal{D}[Pos\_L]$。然而，若 $Neg\_L$ 仅由 $Hard\_L$ 构成会缺少足够的负例让模型学习。因此，我们定义其他负样本 $Other\_L = L - Hard\_L$，最终，负例标签 $Neg\_L$ 由 $Hard\_L$ 和少量的 $Other\_L$ 组成。这种难负样本的构建旨在促进语义近似的模式更频繁地出现在指令中，同时也能在不牺牲性能的情况下**减少训练样本量**（例如，原本需12个指令集的49个schema可减至3个）。

</details>


<details>
  <summary><b>轮询式的指令生成</b></summary>

在完成了上述步骤后，我们得到了最终的schema集合 $L'=Pos\_L + Neg\_L$。在基于schema的信息抽取（IE）指令构造中，schema的作用至关重要，它直接决定了模型需要抽取的信息类型，并且反映了用户的具体需求。传统做法通常将完整的schema一次性整合入指令中，然而，在本研究中，我们采纳了一种**轮询式方法**，限制每次询问的schema数量为 $split\_num$ 个，取值范围在4至6之间。因此 $L'$ 将被分为 $|L'|/split\_num$ 个批次进行询问，每批次询问 $split\_num$ 个schema。即使在评估阶段询问的schema数目与训练时不同，通过轮询机制，我们可以将询问数量平均分散至 $split\_num$ 个，从而缓解泛化性能下降的问题。

</details>



### 2.2IEPile的数据格式

`IEPile` 中的每条数据均包含 `task`, `source`, `instruction`, `output` 4个字段, 以下是各字段的说明

| 字段 | 说明 |
| :---: | :---: |
| task | 该实例所属的任务, (`NER`、`RE`、`EE`、`EET`、`EEA`) 5种任务之一。 |
| source | 该实例所属的数据集 |
| instruction | 输入模型的指令, 经过json.dumps处理成JSON字符串, 包括`"instruction"`, `"schema"`, `"input"`三个字段 |
| output | 输出, 采用字典的json字符串的格式, key是schema, value是抽取出的内容 |


在`IEPile`中, **`instruction`** 的格式采纳了类JSON字符串的结构，实质上是一种字典型字符串，它由以下三个主要部分构成：
(1) **`'instruction'`**: 任务描述, 它概述了指令的执行任务(`NER`、`RE`、`EE`、`EET`、`EEA`之一)。
(2) **`'schema'`**: 待抽取的schema(`实体类型`, `关系类型`, `事件类型`)列表。
(3) **`'input'`**: 待抽取的文本。


[instruction.py](./ie2instruction/convert/utils/instruction.py) 中提供了各个任务的指令模版。


以下是一条**数据实例**：

```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

该数据实例所属任务是 `NER`, 所属数据集是 `CoNLL2003`, 待抽取的schema列表是 ["`person`", "`organization`", "`else`", "`location`"], 待抽取的文本是 "*284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )*", 输出是 `{"person": ["Robert Allenby", "Allenby", "Miguel Angel Martin"], "organization": [], "else": [], "location": ["Australia", "Spain"]}`

> 注意输出中的 schema 顺序与 instruction 中的 schema 顺序一致


<details>
  <summary><b>更多任务的数据实例</b></summary>

```json
{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}

{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}
```

</details>




## 3.使用IEPile训练模型


欲了解如何使用IEPile进行模型的训练与推理，请访问我们的官方GitHub仓库以获取详细教程：[IEPile](https://github.com/zjunlp/IEPile)。

# 一阶段任务
在llm/README_CN.md中有许多示例，我们需要对这些代码进行重新整理
- 首先使用大语言模型对两个数据集的训练数据进行数据增强（可参考LLMICL/README_CN.md/#使用大语言模型进行数据增强）
- 接着使用增强之后的两个数据集的数据使用P-Tuning对我们部署的千问api进行微调（可以参考InstructKGC/README_CN.md/#51p-tuning微调chatglm这一节的代码，但是要各外注意，我们只是黑盒微调）
- 接着使用大语言模型完成指令驱动的知识图谱构建，可以参考LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建
- 最后使用两个数据集的测试数据评估我们的LLM指令驱动效果，评估的时候详细阅读数据集的相关结构