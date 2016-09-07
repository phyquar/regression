# -*- coding: utf-8 -*-
#
#回帰分析ツールまとめ
#順次追加
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

def Draw_curve_by_least_square(x,y,m):
	#x:トレーニングデータのインプット(リスト形式)
	#y:トレーニングデータのアウトプット(リスト形式)
	#m:多項式近似曲線の次数
	#指定した次数の多項式で近似曲線を描く関数
	def create_dataset(x,y):
    		dataset = DataFrame(columns=['x','y'])
    		for i in range(len(x)):
        			dataset = dataset.append(Series([x[i],y[i]], index=['x','y']),ignore_index=True)
    		return dataset


	def resolve(dataset, m):
		t = dataset.y
		phi = DataFrame()
		for i in range(0,m+1):
			p = dataset.x**i
			p.name="x**%d" % i
			phi = pd.concat([phi,p], axis=1)
		tmp = np.linalg.inv(np.dot(phi.T, phi))
		ws = np.dot(np.dot(tmp, phi.T), t)
		def f(x):
			y = 0
			for i, w in enumerate(ws):
				y += w * (x ** i)
			return y
		return (f, ws)

	train_set = create_dataset(x,y)
	#test_set = create_dataset(x,y)
	df_ws = DataFrame()

    	# 多項式近似の曲線を求めて表示

	f, ws = resolve(train_set, m)
	df_ws = df_ws.append(Series(ws,name="M=%d" % m))

	plt.xlim(min(x),max(x))
	plt.ylim(min(y),max(y))
	plt.title("The order=%d" % m)

	# トレーニングセットを表示
	plt.plot(train_set.x, train_set.y, 'bo')

    	# 多項式近似の曲線を表示
	linex = np.linspace(min(x),max(x),101)
	liney = f(linex)
	plt.plot(linex, liney, color='red')
	plt.show()
