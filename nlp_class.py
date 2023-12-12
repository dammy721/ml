#!pip install mecab-python3
#!pip install unidic-lite

import MeCab

def tokenize(text):
    """ 日本語テキストを分かち書きにする """
    tagger = MeCab.Tagger()
    tagger.parse('')  # MeCabのバグを回避
    node = tagger.parseToNode(text)
    words = []
    while node:
        pos = node.feature.split(',')[0]
        if pos in ['名詞', '形容詞', '動詞']:  # 対象とする品詞を選択
            words.append(node.surface)
        node = node.next
    return ' '.join(words)

# データセット
texts = [
    "これはとても良い製品です", "最悪の経験だった", "大変満足しています", "全く役に立たなかった", "非常に楽しい時間を過ごせました",
    "サービスには満足している", "期待はずれだった", "非常に役立つ情報だ", "使い勝手が悪い", "素晴らしい結果に感謝",
    "このアプリは使いにくい", "非常に満足", "問題が多すぎる", "クオリティが高い", "最悪の対応だった"
]
labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: ポジティブ, 0: ネガティブ

texts.extend([
    "非常に効果的な方法だと思う", "まったく期待外れだ", "素晴らしいサービスだ", "不満が多い", "これは私のお気に入りです",
    "品質に問題がある", "非常に有用なアプリ", "操作が複雑すぎる", "おすすめできる商品", "とても不便だ",
    "信頼できる品質", "使い方が難しい", "優れたカスタマーサポート", "まったく役に立たない", "この価格でこの品質は素晴らしい",
    "全然ダメだ", "使い心地が良い", "期待していたよりも悪い", "応答が早くて助かる", "思ったよりも性能が劣る"
])
labels.extend([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# テキストデータの前処理
processed_texts = [tokenize(text) for text in texts]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
import numpy as np

# パイプラインの作成
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
accuracy_scores = []

for train_index, test_index in loo.split(processed_texts):
    X_train, X_test = [processed_texts[i] for i in train_index], [processed_texts[i] for i in test_index]
    y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    accuracy_scores.append(score)

# 平均精度の計算
average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy: {average_accuracy}")
