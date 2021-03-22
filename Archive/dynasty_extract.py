import re

def past_time_concat(loc_list: list):
    '''
    连接误分割的公元前字符串
    '''
    loc_list = loc_list.copy()
    for index, loc_str in enumerate(loc_list):
        if loc_str == '公元前' and '000' in loc_list[index + 1]:
            loc_list[index] = loc_list[index] + loc_list[index + 1]
            loc_list[index + 1] = ''
            print(loc_list[index])
    return loc_list


def time_filter(time_str: str) -> str:
    '''
    过滤无法判断时间的词语
    归类表示现在的词语
    :param time_str:
    :return:
    '''
    if time_str == '' or time_str == ' ':
        return None

    if time_str in [
        '每年', '每日', '当时', '中期', '此后', '当年', '早期', '昔日', '明天', '同年',
        '一时', '过去', '每天', '一日', '其后', '次年', '未来', '后来', '初年', '后期',
        '古代', '初期', '今',
    ]:
        return None

    if time_str in [
        '目前', '现在', '今天', '近年', '现代', '现', '今日', '近代', '此时',
        '近现代', '当代', '当前']:
        return '现代'

    time_str = time_str.replace('时期', '')
    return time_str


def year_filter(time_str: str) -> str:
    '''
    将年转为具体朝代
    :param time_str:
    :return:
    '''

    if '年' not in time_str:
        return time_str

    if time_str.startswith('近'):
        return None
    if '以后' in time_str:
        return None
    if time_str == '2020年前':
        return None

    time_str.replace('1700万，1850年', '1850年')

    pattern_before = re.search('前\S*?(\d{3,4})', time_str)
    pattern_past = re.search('(\d{3,4})\S*?前', time_str)
    pattern_4 = re.search('\d{4}', time_str)
    pattern_3 = re.search('\d{3}', time_str)
    pattern_2 = re.search('公元(\d{2})', time_str)

    if pattern_before:
        time_int = -int(pattern_before.group(1))
    elif pattern_past:
        time_int = 2021 - int(pattern_past.group(1))
    elif '多年' in time_str:
        time_int = None
    elif pattern_4:
        time_int = int(pattern_4.group(0))
    elif pattern_3:
        time_int = int(pattern_3.group(0))
    elif pattern_2:
        time_int = int(pattern_2.group(1))

    else:
        time_int = None

    #     if re.search('\d{4}', time_str):
    #         print(time_str + ' --> ' + re.search('\d{4}', time_str).group(0))
    #         return re.search('\d{4}', time_str).group(0)
    #     elif re.search('\d{1,2}', time_str):
    #         print(time_str + 'X')
    #         return None

    #     if time_int is not None:
    #         print(time_str + ' --> ' + str(time_int))
    #         return str(time_int)
    #     else:
    #         print(time_str + ' X')
    #         return None

    if time_int is None:
        return None

    if time_int >= 1368 and time_int <= 1644:
        return '明朝'
    elif time_int >= 1644 and time_int <= 1912:
        return '清朝'
    elif time_int > 1912:
        return '现代'
    elif time_int >= 1279 and time_int <= 1368:
        return '元朝'
    elif time_int >= 907 and time_int <= 1279:
        return '宋朝'
    elif time_int >= 618 and time_int <= 907:
        return '唐朝'
    elif time_int >= 581 and time_int <= 618:
        return '隋朝'
    elif time_int >= 265 and time_int <= 581:
        return '魏晋南北朝'
    elif time_int >= 220 and time_int <= 265:
        return '三国'
    elif time_int >= -206 and time_int <= 220:
        return '汉代'
    elif time_int >= -221 and time_int <= -206:
        return '秦代'
    elif time_int >= -770 and time_int <= -221:
        return '春秋战国'
    elif time_int >= -1100 and time_int <= -770:
        return '西周'
    elif time_int >= -1600 and time_int <= -1100:
        return '商代'
    elif time_int >= -2100 and time_int <= -1600:
        return '夏代'
    elif time_int <= -2100:
        return '黄帝'
    else:
        raise Exception('No dynasty matched')


def dynasty_filter(time_str) -> str:
    import re

    mapping_dict = {
        '西汉初期': '汉代',
        '西汉': '汉代',
        '明中期': '明朝',
        '唐时': '唐朝',
        '商末': '商代',
        '东汉初': '汉代',
        '当上元': '唐朝',
        '盛唐': '唐朝',
        '元朝末期': '元朝',
        '五代': '宋朝',
        '北魏': '魏晋南北朝',
        '秦汉': ['秦代', '汉代'],
        '元朝初': '元朝',
        '明前期': '明朝',
        '民国初': '现代',
        '明': '明朝',
        '明清时': ['明朝', '清朝'],
        '五代十国': '宋朝',
        '晚唐': '唐朝',
        '隋末': '隋朝',
        '唐中': '唐朝',
        '宋末': '宋朝',
        '唐末': '唐朝',
        '魏晋': '魏晋南北朝',
        '明朝中期': '明朝',
        '先秦': '秦代',
        '战国': '春秋战国',
        '汉朝': '汉代',
        '宋元': ['宋朝', '元朝'],

        '明清': ['明朝', '清朝'],
        '清代': '清朝',
        '元代': '元朝',
        '明代': '明朝',
        '宋代': '宋朝',
        '唐代': '唐朝',
        '隋代': '隋朝',
        '南北朝': '魏晋南北朝',
        '晚清': '清朝',
        '春秋': '春秋战国',
        '元末': '元朝',
        '元初': '元朝',
        '东汉': '汉代',
        '唐': '唐朝',
        '民国': '现代',
        '秦朝': '秦代',
        '唐宋': ['唐朝', '宋朝'],
        '明末': '明朝',
        '明初': '明朝',
        '清初': '清朝',
        '隋唐': ['隋朝', '唐朝'],
        '清末': '清朝',
    }

    if time_str in mapping_dict:
        return mapping_dict[time_str]

    return time_str


def dynasty_select(time_str: str) -> bool:
    return time_str in ['现代', '清朝', '明朝', '唐朝', '宋朝', '元朝', '隋朝', '魏晋南北朝', '春秋战国', '汉代', '秦代', '三国', '黄帝', '夏代', '西周',
                        '商代']


def dynasty_extract(loc_list: list):
    loc_list = loc_list.copy()

    loc_list = past_time_concat(loc_list)

    loc_list = [time_filter(time_str) for time_str in loc_list if time_filter(time_str) is not None]
    loc_list = [year_filter(time_str) for time_str in loc_list if year_filter(time_str) is not None]

    loc_list_copy = loc_list.copy()
    loc_list = []
    for index, time_str in enumerate(loc_list_copy):
        dynasty_result = dynasty_filter(time_str)
        if isinstance(dynasty_result, list):
            loc_list.extend(dynasty_result)
        else:
            loc_list.append(dynasty_result)

    loc_list = [time_str for time_str in loc_list if dynasty_select(time_str)]

    return loc_list

def dynasty_extract_plus_loc(time_list, loc_list):
    pass


if __name__ == '__main__':
    import pandas as pd
    from slab.pickle_util import pickle_to_file, unpickle_from_file

    df = unpickle_from_file('df.pkl')
    loc_list = df['line_time_result'].values
    loc_list = [word for line in loc_list for word in line]

    print(pd.Series(dynasty_extract(loc_list)).value_counts())
