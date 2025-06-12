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

# 一阶段任务
在llm/README_CN.md中有许多示例，我们需要对这些代码进行重新整理
- 首先使用大语言模型进行数据增强（可参考LLMICL/README_CN.md/#使用大语言模型进行数据增强）
- 接着使用增强之后的数据使用P-Tuning对我们部署的千问api进行微调（可以参考InstructKGC/README_CN.md/#51p-tuning微调chatglm这一节的代码，但是要结合qwen自身的特点）
- 接着使用大语言模型完成指令驱动的知识图谱构建，可以参考LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建
- 最后使用两个数据集进行评估，评估的时候分为两个角度，一个是指定schema，一个是使用命名实体识别自行识别schema