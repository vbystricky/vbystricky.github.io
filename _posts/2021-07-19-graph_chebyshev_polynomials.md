---
layout: post
title: Сигналы на графах, свёртки и полиномы Чебышева.
date: 2021-07-19
category: Graph
tags: [Graph, Laplacian, convolution]
use_math: true
published: true
---

Еще раз поговорим про сигналы на графах. И о том как быстро считать свёртки сигнала с использованием полиномов Чебышева. Нужно это нам для того, чтобы
собирать свёрточные сети на графах. Хотя исходно это разбиралось в [1] для навешивание на графы вейвлет преобразований.

<!--more-->

Повторим вначале коротко то, о чем шла речь в [предыдущей статье]({% post_url 2021-06-23-graph_laplacian_clusters_etc %}).

Мы снова работаем с графом $\mathcal G = (\mathcal V, \mathcal E)$, у которого $\mathcal V$ - вершины и $\mathcal E$ - рёбра. Граф у нас взвешенный,
т.е. задана матрица весов $W \in\mathbb R^{N\times N}$ элемент $w_{ij}$, которой не равен нулю, если существует ребро между вершинами $i$ и $j$.
Матрицу $W$ симметричная. Для каждой вершины $i$ определим *степень*:

$$
d_i = \sum_{j=1}^N w_{ij}
$$

и объединим эти величины в диагональную матрицу:

$$
D =
\begin{pmatrix}
d_{1} & & 0\\
& \ddots & \\
0 & & d_{N}
\end{pmatrix}
$$


Используя матрицы $W$ и $D$ можно определить [лапласиан графа]({% post_url 2021-06-23-graph_laplacian_clusters_etc %}#лапласиан-графа):

$$\mathcal L = D - W$$

(пока остановимся на этом варианте лапласиана, хотя к остальным применимы похожие рассуждения).

Показано, что лапласиан неотрицательно определенная симметричная матрица, а значит у него будут вещественные собственные числа 
$0=\lambda_1 \le \lambda_2 \le ... \le \lambda_N$. Найдем для них соответствуюшие собственные вектора:

$$
\left\{u^{(k)} = \left(u^{(k)}_1, u^{(k)}_2,..., u^{(k)}_N\right)^T \right\}_{k=1}^N
$$

причем собственные вектора выберем таким образом, чтобы они составляли ортонормированный базис пространства $\mathbb R^N$. Составим из этих векторов
матрицу $U = (u^{(1)}, u^{(2)},..., u^{(N)})$, в силу ортонормированности векторов эта матрица будет ортогональной, т.е. $U U^T = U^T U = I$.

Наконец лапласиан можно представить в виде: $\mathcal L=U \Lambda U^T$, где $\Lambda = \mathrm{diag}(\lambda)$ - диагональная матрица, составленная из
собственных чисел.

Используя матрицу $U$ мы можем для любого сигнала $f = (f_1, f_2, ..., f_N)^T \in \mathbb R^N$ на графе ($i$-ой вершине приписываем величину $f_i$)
определить прямое: $\tilde f = U^T f$, и обратное: $f = U \tilde f$ преобразования Фурье.

Используя преобразование Фурье, можно задать операцию свёртки $\ast_{\mathcal G}$ сигналов $f$ и $g$ на графе $\mathcal G$:

$$
f \ast_{\mathcal G} g = U \cdot \left((U^T f) \odot (U^T g) \right) = U \cdot \left(\tilde f \odot \tilde g \right)
$$

здесь $\odot$ - покоординатное умножение двух векторов.

Спектральный фильтр в общем виде можно определить при помощи вектора $\tilde \theta \in \mathbb R^N$ и его применение будет выглядеть как:

$$
f' = f \ast_{\mathcal G} \theta = U \cdot {\rm diag}(\tilde \theta) \cdot U^T f
$$

Но в настолько общем виде фильтр зачастую бесполезен (разве что мы хотим вычистить высокие частоты и оставить только низкие или наоборот), поэтому
обычно предлагается задавать фильтр в виде функции $g_{\theta}:\mathbb R_+ \rightarrow \mathbb R$, которая применяется к собственным числам
лапласиана. Т.е. $i$-ая координата вектора $\theta$ задаётся как значение этой функции в $i$-ом собственном числе: $\theta_i = g_{\theta}(\lambda_i)$.
Тогда применение фильтра будет выглядеть как:

$$
f' = f \ast_{\mathcal G} \theta = U \cdot g_{\theta}(\Lambda) \cdot U^T f 
$$

Остаётся проблема с поиском собственных значений и собственных векторов матрицы лапласиана, поскольку вычислительная сложность таких методов
$O(N^3)$ и на достаточно больших графах работа становится практически невыполнимой.

## Операции свёртки на графе и полиномы Чебышева

Чтобы получать приемлимые по времени работы алгоритмы в [1] предлагают интерполировать функцию $g_{\theta}$ полиномами Чебышева.

<div class="sidebar" markdown="1">

### Полиномы Чебышева

Полиномы Чебышева (см., например, [2]) определяются на отрезке $[-1, 1]$, как 

$$T_n(x) = \cos(n \cdot \arccos(x))$$

Основное свойство такого полинома, заключается в том, что на отрезке $[-1, 1]$ из всех полиномом степени $n$ со старшим коэффициентом равным единице,
полином 

$$\tilde T_n(x) = \frac 1 {2^{n-1}} T_n(x)$$

наименее уклоняется от нуля (доказательство есть, например, в [2]).

Так же любые два полинома Чебышева ортогональны по весу:

$$
p(x) = \frac 1 {\sqrt {1 - x^2}}
$$

т.е. 

$$
\int_{-1}^{1} \frac {T_n(x) T_m(x)} {\sqrt {1 - x^2}} dx = 0,\, n\neq m
$$

Произвольную функцию $h \in L_{p(x)}^2\left([-1, 1]\right)$, т.е. функцию, которая удовлетворяет неравенству:

$$\int_{-1}^{1}h^2(x)p(x)dx < \infty$$

можно представить в виде ряда по полиномам Чебышева:

$$h(x) = \frac 1 2 a_0 + \sum_{i=1}^{\infty}a_i T_i(y)$$

коэффициенты ряда, находятся по формуле:

$$
a_i = \frac 2 {\pi} \int_{-1}^1 \frac {T_i(x)h(x)} {\sqrt{1 - x^2}} dx = 
\frac 2 {\pi} \int_{0}^{\pi} \cos(k\phi)h\left(\cos(\phi)\right)d\phi.
$$

Наконец, еще одно важное свойство, которое нам понадобится это *рекуррентное соотношение* с помощью которого можно получать полиномы Чебышева:

$$
\begin{align*}
T_0(x) &= 1,\\
T_1(x) &= x,\\
T_n(x) &= 2xT_{n-1}(x)-T_{n-2}(x)
\end{align*}
$$

</div>

Чтобы использовать полиномы Чебышева для представления функции $g_{\theta}$, которую мы планируем считать в собственных числах лапласиана, нам надо
линейно преобразовать их область определения, отрезок $[-1, 1]$ в отрезок: $[0, \lambda_N]$ положив $y = (x + 1) \frac {\lambda_N} 2$.

Обозначим:

$$
\begin{align*}
\alpha &= \frac {\lambda_N} 2\\
\bar{T}_k(x) &= T_k\left(\frac {x - \alpha} {\alpha}\right),\, \forall x\in [0, \lambda_N]
\end{align*}
$$

Рекуррентное соотношение при $k \ge 2$ перепишется для $\bar{T}_k$ следующим образом:

$$\bar{T}_k(x) = \frac 2 {\alpha} (x-\alpha)\bar{T}_{k-1}(x) - \bar{T}_{k-2}(x).$$

Теперь найдём коэффициенты по формуле:

$$
a_k = \frac 2 {\pi} \int_{0}^{\pi} \cos(k\phi)h\left(\alpha\left(\cos(\phi) + 1\right)\right)d\phi,
$$

тогда $g_{\theta}$ можно представить в виде:

$$
g_{\theta}(x) = \frac 1 2 a_0 + \sum_{k=1}^{\infty} a_k \bar{T}_k(x),\, \forall x\in [0, \lambda_N] 
$$

Вернемся к свёртке сигнала. Мы уже показали в [прошлый раз]({% post_url 2021-06-23-graph_laplacian_clusters_etc %}#свёртка-сигналов-на-графе), что для
многочлена $h(x) = \sum_{k=0}^K a_k x^k$ свёртку вида:

$$
f' = U \cdot h(\Lambda) \cdot U^T f 
$$

можно в силу ортогональности матрицы $U$, переписать в виде:

$$
f' = h(\mathcal L) f 
$$

здесь $h(\mathcal L) = \sum_{k=0}^K a_k {\mathcal L}^k$.

Для многочленов Чебышева мы можем записать, используя рекуррентное соотношение:

$$
\bar{T}_k(\mathcal L)f = \frac 2 {\alpha} (\mathcal L - \alpha I) \left(\bar{T}_{k-1}(\mathcal L)f\right) - \bar{T}_{k-2}(\mathcal L)f,
$$

или, обозначив действие $k$-го многочлена Чебышева от лапласиана на cигнал $f$ как $y^{(k)} = \bar{T}_k(\mathcal L)f$ получим:

$$
\begin{align*}
y^{(0)} &= f,\\
y^{(1)} &= \frac 1 {\alpha} (\mathcal L - \alpha I) f,\\
y^{(k)} &= \frac 2 {\alpha} (\mathcal L - \alpha I)y^{(k-1)}  - y^{(k-2)},\, \forall k > 2
\end{align*}
$$

Таким образом, чтобы вычислить результат действия $k$-го многочлена Чебышева на сигнал $f$, зная результат для $(k-1)$-го и $(k-2)$-го нам необходимо
выполнить только одно матричное умножение лапласиана на вектор. Причем матрица $\mathcal L$ сильно разрежена и умножение её на вектор имеет сложность
$O(\vert\mathcal E\vert)$, линейно зависящую от числа рёбер графа. Теперь, если мы в представлении рядом функции $g_{\theta}$ ограничимся только
первыми $K$ степенями, то используя рекуррентное соотношение, вычисление:

$$
f' = f \ast_{\mathcal G} \theta \approx \frac 1 2 a_0 I f + \sum_{k=1}^{K} a_k \bar{T}_k(\mathcal L)f
$$

будет иметь сложность $O\left(K\vert\mathcal E\vert\right)$.

Надо отметить (мы это показывали [ранее]({% post_url 2021-06-23-graph_laplacian_clusters_etc %}#свёртка-сигналов-на-графе)), что $i$-ая координата
$f'_i$ будет зависить от значений сигнала $f$ только в вершинах, которые отстают от вершины $i$ не дальше чем на $K$ рёбер. 

---

### Литература

1. *D. K. Hammond, P. Vandergheynstb, R. Gribonval, "Wavelets on Graphs via Spectral Graph Theory",
[arXiv:0912.3848](https://arxiv.org/abs/0912.3848), 2009*

2. *Натансон И.П. "Конструктивная теория функций", 1949*
