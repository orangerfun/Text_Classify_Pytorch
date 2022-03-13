# Text-Classify-Pytorch
构建基于BERT等预训练模型和torch框架的文本分类的架构，使得能够达到开箱换数据集就能使用的效果

# 1. quick start
为了方便调试，并没有使用shell脚本的方式传参，而是直接在`finetuning_argparse.py`文件中给每个参数默认都赋值了

* 训练
  
```python
python run_classify.py
```

* predict <br>
将`run_classify.py`文件中 line 447 的main函数传入参数`[False, False, True]`

# 2. 文件及目录介绍
* `callback` 目录存放了一些优化器等
* `losses` 目录存放了一些损失函数
* `outputs` 目录存放保存的模型及预测输出文件
* `pretrained_model` 目录存放预训练模型，如bert
* `processed_data` 存放处理后能被程序加载的统一数据格式的数据
* `processors` 目录存放数据处理相关代码
* `tools` 存放utils
* `build_dataset.py` 构建数据集
* `classify_model.py` 模型
* `finetuning_argparse.py` 参数
* `run_classify.py` 主函数，项目程序入口，包括训练，预测，测试等
  
# 3. 数据说明
原始数据需要处理成程序能够读取的格式，并存放到`processed_data`目录下，然后直接运行`run_classify.py`文件，本项目即可开箱即用。

具体的数据统一格式如下：
* 单条文本分类 task_name = "classify"<br>
是一个json文件，每一行为一个字典，字典的key值只能是`question`和`label`
```json
{"question": "中集集团09年净利润下降超三成新浪财经讯 3月23日晚间消息，中集集团周一晚间发布09年业绩，2009 年公司实现营业收入204.76 亿元，同比下降56.74%；归属于母公司股东的净利润9.59 亿元，同比下降31.84%，基本每股收益0.36元。对于业绩为何呈现下降趋势，年报显示，因2009年全球经济出现严重衰退，主要经济体消费萎靡和出口萎缩并存，导致国际贸易量及航运业需求明显下滑。据年报数字，2009年全球集装箱贸易量、中国规模以上港口集装箱吞吐量均出现负增长，同比分别下降9.5%和5.8%。物流、能源化工等行业也不同程度受到经济衰退影响。但据公司分析，受益于欧美经济的复苏，“补库存”需求的持续，2010 年中国外贸出口形势将随之明显好转，集装箱市场需求将进一步回升。中集集团今日收报14.68元，上涨0.02元，涨幅达0.14%。(俊萍 发自深圳)\n", "label": "股票"}
{"question": "七旬疑犯猝死看守所 家属称尸体脸部带伤庐江县龙桥镇76岁老人谢富常因打伤81岁的亲哥哥，日前被庐江县检察机关批准逮捕。3月26日，这位老人在看守所内发生意外，送到医院后很快被宣告死亡。对于老人的死因，庐江县警方认为可能系猝死。目前，庐江县检察院已介入调查。老兄老弟大动干戈4月4日晚，记者来到庐江县龙桥镇新建村，见到了谢富常老人的小儿子谢贵生。得知父亲死讯后，他刚刚从上海赶回老家。谢贵生说，他和哥哥都在上海做生意，母亲也在上海，庐江的家中只有父亲谢富常一人。由于谢富常和81岁的二哥谢某住得不远，双方因琐事产生一些矛盾。今年2月21日，两位老人之间矛盾升级，谢富常将谢某打得昏迷不醒，后者很快被送往合肥市一家大医院急救。远在上海的谢贵生得知消息后，于2月22日上午从上海赶到合肥，看望二伯父谢某。当天下午，他赶到庐江县龙桥镇家中时，得知父亲已被警方带走。几天后，他接到通知，父亲因故意伤害罪被批捕。“两位老人究竟因何事而大动干戈，由于双方当事人一个被逮捕，另一个正处于昏迷状态，因此很难知道详情。”谢贵生说，由于他没法见到父亲了解情况，只能再次回到上海。死讯来得十分突然3月26日，正在上海与人谈生意的谢贵生接到老家一位村干部的电话，称谢富常死了。惊闻父亲的死讯，谢贵生立即赶回庐江老家。3月27日上午，庐江县检察院、公安局、看守所负责人以及政府部门有关人士与谢贵生会面，进行了交流。公安机关表示，经初步调查，谢富常可能是在看守所内不小心摔倒而猝死。3月27日中午，谢贵生在庐江县殡仪馆见到了父亲的遗体，他发现父亲的脸上带着伤痕，裤子上全是粪便。谢贵生对父亲的遗体进行了拍照。昨天，谢贵生一边向记者出示照片一边说：“我父亲生前身体很好，就是脾气倔了一点，不知道为什么就这么匆匆地走了。”检察院已介入调查4月4日，记者就此事采访了庐江县公安局和巢湖市公安局有关人士，他们证实谢富常老人确实在看守所内发生意外后死亡，但对于此事的详细情况，他们表示不便透露。此后，记者又通过谢贵生与庐江县检察院一位副检察长取得联系，这位副检察长称，目前多个部门都在认真调查此事，不过有关情况尚不能对外公开。这位副检察长表示，自从外地发生“躲猫猫”事件之后，各地对看守所内发生的意外事件都十分重视，“在处理这类事件上，谁都不敢遮掩。我们肯定会公平公正，依法办事。”本报记者 何雪峰 金学永\n", "label": "社会"}
```

* 两条文本相似度 task_name="sim" <br>
是一个json文件，每一行为一个字典，字典的key值只能为 `question` `relationship`和`label`; question和relationship分别表示要计算相似度的两条文本
```json
{"question": "龙天动地有多少页？", "relationship": "发起时间", "label": "0"}
{"question": "司马村的土地面积有多少？", "relationship": "主要产品", "label": "0"}
```

# 4. 模型和数据集下载
本文使用了`THUCNews`数据集进行多分类的训练
使用`kgClue`数据集进行二分类模型的训练

模型方面尝试了`rbt3`和`roberta_wwm_ext`

模型及数据集下载链接：https://pan.baidu.com/s/1bnk-2oVqCdH1eAjTdqky2A 
提取码：ifrm

**效果**
使用rbt3模型：两个文本匹配任务 accuracy: 0.778
             新闻文本（单条文本）分类任务: accuracy: 0.92+
