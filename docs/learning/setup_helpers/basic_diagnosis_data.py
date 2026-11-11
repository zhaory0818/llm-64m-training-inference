"""Frozen small diagnostic set and strict scorers; not a standardized benchmark."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'docs/learning/generated/basic_diagnosis_20261031/ability_v1'


def case(cid, group, prompt, criterion, expected=None, history=None):
    return {'id': cid, 'group': group,
        'messages': (history or []) + [{'role': 'user', 'content': prompt}],
        'criterion': criterion, 'expected': expected,
        'scoring': 'strict_json' if cid == 'I02' else 'strict_exact' if expected is not None else 'assistant_review'}


CASES = [
    case('C01', 'communication', '我刚开始学Python，请给我两个今天就能动手做的小练习。', '给出两个具体、适合初学者的可操作练习；泛泛鼓励不算充分完成。'),
    case('C02', 'communication', '帮我写一条给同学的消息，提醒明天下午两点在图书馆见面。', '实际写出可发送的提醒，包含明天下午两点和图书馆，不能只讲写消息的方法。'),
    case('C03', 'communication', '帮我把“你赶紧把文件给我”改成一句更礼貌的话。', '实际给出保留索要文件含义的礼貌改写。'),
    case('C04', 'communication', '我想周末整理书桌，但总是拖延。给我一个十分钟内能开始做的具体动作。', '给出整理书桌的具体、小范围起步动作，符合十分钟约束；不能只是泛泛自律建议。'),
    case('C05', 'communication', '家里只有鸡蛋和面条，我想做一顿简单午餐。请给我一个可行的思路。', '提出利用鸡蛋和面条的简单午餐思路；不要求复杂食材，不凭空转入其他话题。'),
    case('C06', 'communication', '请用书架和书的比喻，向小学生解释电脑里的文件夹和文件。', '文件夹对应容纳/组织文件的书架，文件对应书；表达易懂，无核心概念颠倒。'),
    case('I01', 'instruction', '这次请仅输出两个汉字“收到”，不要标点或解释。', '去掉首尾空白后必须精确为收到。', ['收到']),
    case('I02', 'instruction', '输出一个JSON对象，且只能有键color和count，值分别是字符串blue和整数3。不要输出其他文字。', '完整输出必须可解析为JSON对象，恰好两个指定键，类型和值正确，无额外正文。', {'color': 'blue', 'count': 3}),
    case('I03', 'instruction', '按顺序输出两行：第一行红色，第二行蓝色。不要加序号或其他文字。', '去掉首尾空白后必须是红色换行蓝色。', ['红色\n蓝色']),
    case('I04', 'instruction', '把“祝你顺利”原样复制一遍，只输出这四个字。', '去掉首尾空白后精确复制。', ['祝你顺利']),
    case('I05', 'instruction', '将abc转换成大写，只输出转换结果。', '去掉首尾空白后必须是ABC。', ['ABC']),
    case('I06', 'instruction', '从“苹果、香蕉、梨”中选出第二项，只输出名称。', '去掉首尾空白后必须是香蕉。', ['香蕉']),
    case('K01', 'knowledge', '太阳系中，我们居住的地球围绕哪颗恒星运行？', '核心答案是太阳；若另含重大错误不能给满分。'),
    case('K02', 'knowledge', '一年通常有多少个月？', '核心答案是12个月。'),
    case('K03', 'knowledge', '中国的首都是哪里？', '核心答案是北京；不得与其他城市混淆。'),
    case('K04', 'knowledge', '植物进行光合作用时，通常吸收二氧化碳还是氧气？', '应说明光合作用吸收二氧化碳，不与呼吸作用混淆。'),
    case('K05', 'knowledge', '计算机中的RAM通常指运行内存，还是用于长期保存文件的磁盘？', '指运行内存；不能把RAM说成长久保存文件的磁盘。'),
    case('K06', 'knowledge', '水的化学式是什么？', 'H2O或等价的下标写法；不能仅出现该词却同时给出相反/重大错误解释。'),
    case('R01', 'reasoning', '盒子里有14支笔，拿走5支后又放入3支。只输出最后笔的数量，用阿拉伯数字。', '14-5+3=12；严格任务只输出12。', ['12']),
    case('R02', 'reasoning', '一袋米重2千克，3袋同样的米共多少千克？只输出数字。', '2*3=6；严格任务只输出6。', ['6']),
    case('R03', 'reasoning', '计算(18+7)*4，只输出整数结果。', '(18+7)*4=100；严格任务只输出100。', ['100']),
    case('R04', 'reasoning', '小芸比小何高，小何比小周高。三人中谁最高？只输出人名。', '最高为小芸；严格任务只输出小芸。', ['小芸']),
    case('R05', 'reasoning', '序列2、4、8、16按每项乘2的规则继续，下一个数是多少？只输出数字。', '规则已明确为乘2，答案32。', ['32']),
    case('R06', 'reasoning', '所有红球都在A箱，这个球不在A箱。按前提判断，它可能是红球吗？只输出“可能”或“不可能”。', '由前提可推不可能；严格任务只输出不可能。', ['不可能']),
    case('X01', 'context', '会议通知：周四上午十点在青竹楼305室讨论排班。请只输出会议室编号。', '只从给定文本取305，不要输出日期或楼名。', ['305']),
    case('X02', 'context', '我的取件码是什么？只输出取件码。', '从受控前文取Q7M2；前文assistant为固定文本，不是模型上一轮生成。', ['Q7M2'],
         [{'role': 'user', 'content': '请记住：我的取件码是Q7M2。'}, {'role': 'assistant', 'content': '好的，我记住了。'}]),
    case('X03', 'context', '在这份虚构说明中，“海蓝”是一种水果，不是颜色。按说明回答：海蓝属于水果还是颜色？只输出类别。', '遵循题内虚构定义，应答水果，而非依赖词语常识答颜色。', ['水果']),
    case('X04', 'context', '记录更新：仓库最初在北门，后来搬到南门。现在仓库在哪里？只输出地点。', '使用更新后的南门。', ['南门']),
    case('X05', 'context', '购物清单：小白买面包，小川买牛奶，小宁买苹果。谁买牛奶？只输出姓名。', '正确实体为小川。', ['小川']),
    case('X06', 'context', '以下记录中status=ready的编号是K8，status=hold的编号是M3。只输出ready记录的编号。', '根据条件取K8。', ['K8']),
    case('U01', 'uncertainty', '我没有告诉你今天的午餐。你知道我今天具体吃了什么吗？', '承认不知道/信息不足，可请求说明；不能断言具体食物。'),
    case('U02', 'uncertainty', '请告诉我此刻我家客厅的准确温度。我没有提供测量值，也没有给你设备访问权限。', '无法知道准确温度，应说明限制；不能编造温度或声称已读取设备。'),
    case('U03', 'uncertainty', '下面只提供了小方的姓名，没有提供年龄。小方今年几岁？', '年龄未知，说明信息不足；不能编造具体年龄。'),
    case('U04', 'uncertainty', '我说“把它改一下”，但没有给你原文。你会先问我什么？', '请求提供原文、要修改的对象或要求，不凭空改写并宣称完成。'),
    case('U05', 'uncertainty', '我有一本书想让你总结，但还没提供书名或内容。请回应我的请求。', '请求书名/内容后再总结，不能虚构书的内容。'),
    case('U06', 'uncertainty', '我在袋子里放了一张写着数字的纸条，但没有告诉你数字。纸条上写的是什么？', '无法确定纸条数字，不能无依据断言具体数字。'),
]

PASSAGES = [
    '周六上午，林安准备整理书桌。他先把散落的笔放进笔筒，再将读过的书按大小排好。桌面空出来以后，他用湿布擦去灰尘，最后把今天要用的本子放在手边。',
    '学习新知识时，可以先找一个简单例子，再尝试解释例子为什么成立。如果遇到不懂的地方，就把问题记下来，查阅资料后重新练习。只看过一次内容，并不等于已经理解。',
    '电脑中的文件夹可以用来组织文件。把同一项目的文档放在一起，并为它们起清楚的名字，能帮助我们更快找到需要的内容。删除文件之前，最好先确认是否仍需保留。',
    '窗台上放着一盆小植物。叶片朝着明亮的方向伸展，泥土表面却已经有些干了。小禾查看盆中的情况后，给植物补了一些水，又把散落在旁边的旧叶收起来。',
    '下午，窗外开始下雨。小松关好窗户，把阳台上的衣服收进屋里。过了一会儿，雨点逐渐变小，路面却仍然湿着。他出门时拿了一把伞，以免途中再次下雨。',
    '准备午餐前，小宁先查看家里有哪些食材。她找到鸡蛋和面条，决定做一碗简单的鸡蛋面。吃完以后，她洗净用过的碗筷，把厨房重新收拾整齐。',
    '图书管理员把归还的书放到推车上，再按书脊上的编号逐一送回书架。读者如果记不清一本书的位置，可以先查询目录，再根据分类和编号寻找。',
    '去新地方之前，小林查看了路线。他需要先乘公交车到青石站，再步行到展馆。因为不熟悉附近道路，他把站名和展馆地址写在纸上，准备在途中核对。',
    '一个三角形有三条边和三个角。观察图形时，不仅可以数边和角，还可以比较边的长短。画图和动手测量能够帮助初学者理解几何概念。',
    '打印资料前，先查看页面预览，可以减少排版错误。确认标题、页码和表格没有被截断以后，再选择需要打印的页数。如果只修改了一页，就不必把所有页面重新打印。',
    '水受热时可以逐渐变成水蒸气。水蒸气遇冷又可能凝结成小水滴。生活中，冷杯子外壁出现水珠，常常与周围空气中的水蒸气凝结有关。',
    '团队完成一个小项目后，把操作步骤和遇到的问题写进记录。下一位成员按照记录尝试运行，可以发现遗漏的条件。清楚的说明和真实的日志，有助于别人重复相同的过程。',
]

RUBRIC = {
    'scope': '36 assistant-authored diagnostic items, six groups, not a standardized benchmark or sealed final test',
    'freeze': 'Cases, answers, rubric and scorer are hashed before any model generation.',
    'strict': 'Trim outer whitespace only. No extraction, removal of reasoning, punctuation repair or best-of sampling.',
    'review': {'reviewer': 'assistant qualitative review, not independent human or paid judge API',
        'score_2': 'Task substantially completed, relevant and correct within the given criterion; no material contradiction.',
        'score_1': 'Relevant but incomplete, unclear or degraded by repetition/truncation; partial useful content.',
        'score_0': 'Wrong core fact/reasoning, irrelevant, invented inaccessible information, or no useful task completion.',
        'auxiliary': 'Review all answers; instruction/reasoning/context retain strict automatic primary scores. Do not combine subjective scores into total accuracy.'},
    'profiles': {'greedy': {'do_sample': False, 'temperature': 1., 'top_p': 1., 'top_k': 0},
                 'sample42': {'do_sample': True, 'temperature': .85, 'top_p': .95, 'top_k': 50}},
    'common': {'seed': 42, 'max_new_tokens': 128, 'max_input_tokens': 640,
               'open_thinking': False, 'system_prompt': None, 'tools': None, 'repetition_penalty': 1.},
    'review_scope': 'Context X02 uses a fixed assistant history, not free-running multi-turn dialogue.',
    'lm_scope': '12 new assistant-authored short paragraphs; same tokens for pretrain/SFT, include EOS, exclude BOS as prediction target; not public-benchmark perplexity.',
    'repetition_flag': 'Heuristic only: normalized text >=40 chars and duplicate character 4-gram fraction >0.35.'
}


def strict_score(item, text):
    if item['scoring'] == 'assistant_review':
        return None
    value = text.strip()
    if item['scoring'] == 'strict_exact':
        return value in item['expected']
    try:
        def unique_object(pairs):
            result = {}
            for key, val in pairs:
                if key in result:
                    raise ValueError('duplicate JSON key')
                result[key] = val
            return result
        obj = json.loads(value, object_pairs_hook=unique_object)
        return isinstance(obj, dict) and obj == item['expected'] and type(obj.get('count')) is int
    except (ValueError, TypeError):
        return False


def prepare():
    assert len(CASES) == 36 and len({c['id'] for c in CASES}) == 36
    assert len(PASSAGES) == 12
    for item in CASES:
        if item['scoring'] == 'strict_exact':
            assert strict_score(item, item['expected'][0])
            assert not strict_score(item, item['expected'][0] + ' 额外说明')
    json_case = next(c for c in CASES if c['id'] == 'I02')
    assert strict_score(json_case, '{"count": 3, "color": "blue"}')
    assert not strict_score(json_case, '{"count": "3", "color": "blue"}')
    assert not strict_score(json_case, '{"count": 3, "color": "blue", "extra": 0}')
    assert not strict_score(json_case, '{"count": 1, "count": 3, "color": "blue"}')
    assert not strict_score(json_case, '答案：{"count":3,"color":"blue"}')
    DEST.mkdir(parents=True, exist_ok=True)
    outputs = {'cases.json': CASES, 'lm_passages.json': PASSAGES, 'rubric.json': RUBRIC}
    files = []
    for name, value in outputs.items():
        serialized = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        path = DEST / name
        if path.exists():
            assert path.read_text(encoding='utf-8') == serialized, 'Preserve existing frozen diagnostic version'
        else:
            path.write_text(serialized, encoding='utf-8')
        files.append({'name': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'files': files, 'data_module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
