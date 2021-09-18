from .base import *


def draw_sentiment(df: pd.DataFrame,
                   topn: int = 5,
                   topic_column: str = 'bert_topic',
                   time_column: str = 'timestamps',
                   sentiment_column: str = 'sentiment',
                   save_dir: Union[Path, str] = '.',
                   ):
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    plt.clf()

    df['sentiment_abs'] = df[sentiment_column].abs()

    # 去除-1主题，选取前n个主题
    top_count_topic = df.groupby(topic_column).size().sort_values(ascending=False).index.values
    top_count_topic = top_count_topic[top_count_topic != -1]
    top_count_topic = top_count_topic[: topn]

    df.groupby([topic_column, time_column])[sentiment_column].count().unstack(0)[
        top_count_topic
    ].plot(figsize=(20, 5))
    plt.title(f'前{topn}数量主题数量随时间变化')
    plt.savefig(save_dir_path / 'count_by_time.png')

    df.groupby([topic_column, time_column])[sentiment_column].mean().unstack(0)[
        top_count_topic
    ].plot(figsize=(20, 5))
    plt.title(f'前{topn}数量主题情感平均值随时间变化')
    plt.savefig(save_dir_path / 'sentiment_by_time.png')

    df.groupby([topic_column, time_column])['sentiment_abs'].mean().unstack(0)[
        top_count_topic
    ].plot(figsize=(20, 5))
    plt.title(f'前{topn}数量主题情感绝对值平均值随时间变化')
    plt.savefig(save_dir_path / 'sentiment_abs_by_time.png')

    df.boxplot(column=sentiment_column, by=time_column, figsize=(20, 10))
    plt.savefig(save_dir_path / 'sentiment_boxplot_over_timestamp.png')
    df.boxplot(column='sentiment_abs', by=time_column, figsize=(20, 10))
    plt.savefig(save_dir_path / 'sentiment_abs_boxplot_over_timestamp.png')
    df.boxplot(column=sentiment_column, by=topic_column, figsize=(20, 10))
    plt.savefig(save_dir_path / 'sentiment_boxplot_over_bert_topic.png')
    df.boxplot(column='sentiment_abs', by=topic_column, figsize=(20, 10))
    plt.savefig(save_dir_path / 'sentiment_abs_boxplot_over_bert_topic.png')


def draw_word_freq_over_time(
        df: pd.DataFrame,
        axes_per_row: int = 4,
        save_dir: Union[Path, str] = '.',
):
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    timestamps = df['timestamp'].unique()
    topic_list = df['topic'].unique()
    for topic in topic_list:
        df_topic = df[df['topic'] == topic]

        max_freq = df_topic['frequency'].max()
        topic_word_list = df_topic['word'].unique()

        # 画图
        subplot_num = len(topic_word_list)
        subplot_col = axes_per_row
        subplot_row = math.ceil(float(subplot_num) / subplot_col)
        plt.figure(figsize=(4 * subplot_col, 4 * subplot_row))
        plt.suptitle(
            f'主题ID：{topic}，共{len(topic_list)}个主题，所显示关键词数量：{subplot_num}')

        # for word_index, (word, prob_list) in enumerate(word_stat.items()):
        #     pass

        for word_index, word in enumerate(df_topic['word'].unique()):
            df_word = df_topic[df_topic['word'] == word]

            plt.subplot(subplot_row, subplot_col, word_index + 1)

            x = df_word['timestamp'].values
            y = df_word['frequency'].values

            plt.plot(
                x, y, marker='o', label=word
            )
            plt.xticks([*range(0, timestamps[-1], 2), timestamps[-1]])
            plt.ylim(0, max_freq)
            plt.legend()

        plt.savefig(save_dir_path / f'word_freq_over_time_topic{topic}.png')
