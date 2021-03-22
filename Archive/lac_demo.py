import paddlehub as hub
import pandas as pd
from slab.pickle_util import pickle_to_file, unpickle_from_file

lac = hub.Module(name="lac")

# 保留每行的分词结果，只保存人名、地名、时间
# line_cut_result = []

line_total_result = []


def is_tag_preserved(tag):
    return tag in [
        'PER', 'LOC', 'ORG', 'TIME', 'nr', 'ns', 'nt', 't',
        'a', 'ad', 'j', 'l', 'n', 'ns', 'nt', 'nz', 'v', 'vd', 'vn'
    ]


def lac_text(text_path='text.txt') -> pd.DataFrame:
    with open(text_path, encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            cut_result = lac.cut(text=[line], return_tag=True)[0]
            word = cut_result['word']
            tag = cut_result['tag']

            line_result = [[word, tag] for word, tag in zip(word, tag) if is_tag_preserved(tag)]

            line_loc_result = [word for word, tag in line_result if tag in ['PER', 'LOC', 'ORG', 'nr', 'ns', 'nt']]
            line_time_result = [word for word, tag in line_result if tag in ['TIME', 't']]

            line_loc_result_str = ', '.join(line_loc_result)
            line_time_result_str = ', '.join(line_time_result)

            line_total_result.append({
                'text': line,
                'line_loc_result': line_loc_result,
                'line_time_result': line_time_result,
                'line_loc_result_str': line_loc_result_str,
                'line_time_result_str': line_time_result_str,
                'line_result': line_result
            })

            # line_cut_result.append(line_result)

    df = pd.DataFrame(line_total_result)

    return df


if __name__ == '__main__':
    df = lac_text()
    pickle_to_file(df, 'df.pkl')

    df = unpickle_from_file('df.pkl')
    df.to_html('df.html', columns=['text', 'line_loc_result', 'line_time_result'])
