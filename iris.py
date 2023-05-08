# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

#データセットの読込
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#目標値
df['target'] = iris.target

#目標値を数値から花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'


#予測モデル構築

x = iris.data[:, [0, 2]]
y = iris.target

#ロジスティクス回帰
clf = LogisticRegression()
clf.fit(x, y)


# サイドバー（入力画面）
st.sidebar.header('Input Features')

sepalValue = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

### インプットデータ（1行のデータフレーム）
record = {'data': 'data','sepal length (cm)': sepalValue,'petal length (cm)': petalValue}
### データフレームを初期化
df_1 = pd.DataFrame()
### 辞書をデータフレームへ追加
### ignore_index=True 元の行番号のまま追加が行われますが、ignore_index=True パラメータを指定することで、新たな行番号を割り当てる。
value_df = pd.concat([df_1, pd.DataFrame(record, index=[0])], ignore_index=True)
### data という値を index 名に設定するため
value_df.set_index('data',inplace=True)

# 入力値の値
st.write(value_df)

# 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'], index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です！')