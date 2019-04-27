---
layout: post
title: Tensorflow
date: 2019-04-27
category: tensorflow
tags: [tensorflow]
use_math: true
---

[Tensorflow](http://tensorflow.org) - это набор инструментов машинного обучения с открытым исходным кодом.

<!--more-->

Tensorflow (*TF*) в качестве данных оперирует с *тензорами* правда в более узком смысле, чем тот который в этот термин вкладывается в математике. В TF
тензор - это просто название для многомерного массива данных. Основное предназначение TF это тренировка и вывод нейронных сетей, и для этого
реализовано масса функционала. В том числе различные методы оптимизации, методы работы с изображениями, разные слои для свёрточных сетей и т.п.

Любые вычисления в TF разбиваются на две логические части. Первая это создание *графа вычислений*, вторая - создание сессии и собственно запуск
вычислений на графе. 

## Граф вычислений.

Чтобы сложить $2 + 3$ при помощи TF нам надо первым делом добавить в граф вычислений нужные операции:

```python
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
sum = tf.add(a, b)
print(sum)
```

Этот код выведит в консоль, что-то вроде:

```
Tensor("Add:0", shape=(), dtype=int32)
```

Если начать разбираться, то окажется, что данный код создал в графе вычислений три вершины. Две для констант $a = 2$ и $b = 3$, и третью для
результата операции сложения. А еще было создано два ребра из вершин с константами в вершину с операцией. Именно ссылку на вершину, представляющую
результат операции сложения мы и поместили в переменную $sum$. Т.е. это не сумма $2 + 3$, это ссылка на вершину в графе вычислений, в которой лежит
тензор, с результатом операции сложения.

`Tensor("Add:0", shape=(), dtype=int32)` говорит нам о том, что эта вершина получила автоматически сгенерированное название *"Add:0"*, и представляет
собой тензор ранга ноль (*shape = ()*), содержащий целые числа (*dtype=int32*). Проще говоря 32 битное целое число, как мы и расчитывали.

Если мы хотим вместо целых чисел использовать числа с плавающей точкой, то можно это явно указать при создании констант:

```python
import tensorflow as tf
a = tf.constant(2, dtype=tf.float32)
b = tf.constant(3, dtype=tf.float32)
sum = tf.add(a, b)
print(sum)
```

В результате получим:

```
Tensor("Add:0", shape=(), dtype=float32)
```

Или определить тип констант неявно:

```python
import tensorflow as tf
a = tf.constant(2.)
b = tf.constant(3.)
sum = tf.add(a, b)
print(sum)
```

Результат тот же:

```
Tensor("Add:0", shape=(), dtype=float32)
```

На самом деле сами константы тоже не обязательно объявлять явно, код:

```python
import tensorflow as tf
sum = tf.add(2., 3.)
print(sum)
```

делает ровно тоже самое, что и предыдущий.

## Сессия вычислений

После того как граф создан, можно переходить к получению результатов вычислений. Для этого необходимо создать сессию и в рамках этой сессии получить
значение нужного тензора. Например, вот так:

```python
import tensorflow as tf
sum = tf.add(2., 3.)
with tf.Session() as sess:
    print(sess.run(sum))
```

или вот так:

```python
import tensorflow as tf
sum = tf.add(2., 3.)
with tf.Session() as sess:
    print(sum.eval())
```

Второй вариант мне нравится не очень, так что в дальнейшем я буду использовать `sess.run()`. Вызов любого из этих двух методов приведёт к тому, что TF
попытается вычислить значение тензора в запрашиваемой вершине. При этом по необходимости будут рекурсивно вычислены тензоры в тех вершинах, на которые
ссылается нужная. В нашем примере, промежуточных вершин с вычислениями нет, и для суммирования TF просто подтянет константы `2.` и `3.` (из вершин где
они лежат как константы).

Можно получить и значения сразу нескольких тензоров, просто передав в `sess.run()` их список, например:

```python
import tensorflow as tf
sum1 = tf.add(2., 3.)
sum2 = tf.add(3., 2.)
with tf.Session() as sess:
    print(sum.run([sum1, sum2]))
```

## Заполнитель (*placeholder*)

Понятно, что на одних константах далеко не уедешь. Допустим нам надо вычислить функцию $f(x) = x^2 + 6x + 9$ при трёх различных $x = 1, 2, 3$. Можно
написать код вроде такого:

```python
import tensorflow as tf
a = tf.constant(1., dtype = tf.float32)
b = tf.constant(6., dtype = tf.float32)
c = tf.constant(9., dtype = tf.float32)

X1 = tf.constant(1., dtype = tf.float32)
Y1 = a * X1 * X1 + b * X1 + c

X2 = tf.constant(2., dtype = tf.float32)
Y2 = a * X2 * X2 + b * X2 + c

X3 = tf.constant(3., dtype = tf.float32)
Y3 = a * X3 * X3 + b * X3 + c
with tf.Session() as sess:
    print(sess.run(Y1))
    print(sess.run(Y2))
    print(sess.run(Y3))
```

И получить результат. Но выглядит этот код крайне странно, добавляет в граф вычислений слишком много лишних операций, и при необходимости вычислить
ту же функцию в новой точке потребует добавлять ещё.

В TF для передачи снаружи в граф данных необходимых для вычислений предусмотрен заполнитель (*placeholder*). При создании графа вычислений, мы
добавляем вершину заполнитель, из которой операции планируют получать данные:

```python
import tensorflow as tf
a = tf.constant(1., dtype = tf.float32)
b = tf.constant(6., dtype = tf.float32)
c = tf.constant(9., dtype = tf.float32)
    
X = tf.placeholder(dtype = tf.float32, shape = (), name = "X")
Y = a * X * X + b * X + c
```

Для этой вершины надо определить тип данных (в нашем случае `tf.float32`), ранг и размерности тензора (в нашем случае мы планируем передавать скаляр,
поэтому `shape = ()`), можно задать еще и название, но это не обязательно (хотя с моей точки зрения крайне желательно).

Теперь, если мы как и раньше запустим вычисление значения `Y` кодом:

```python
... 
Y = a * X * X + b * X + c
with tf.Session() as sess:
    print(sess.run(Y))
```

то получим ошибку. Ошибка эта, как это обычно бывает в питоне, а особенно когда работаем с TF, будет на пару экранов текста, но внутри можно будет
отыскать:

```
... You must feed a value for placeholder tensor 'X' ...
```

Что вполне логично, чтобы вычислить `Y` нужно значение `X`, а его не передали. 

> #### Замечание
> Вот кстати почему крайне полезно определять названия для заполнителей. TF, конечно, присваивает всему, что добавляется в граф названия
> автоматически, но понять в большом графе какая именно операция/заполнитель/переменная получила вот это название весьма не просто. А у сообщения    
> `... You must feed a value for placeholder tensor 'Placeholder' with dtype float ...`
> информативность сильно так себе.

Передать данные в заполнитель, можно используя параметр `feed_dict` метода `sess.run`. Итак поправив код:

```python
import tensorflow as tf
a = tf.constant(1., dtype = tf.float32)
b = tf.constant(6., dtype = tf.float32)
c = tf.constant(9., dtype = tf.float32)
    
X = tf.placeholder(dtype = tf.float32, shape = (), name = "X")
Y = a * X * X + b * X + c

with tf.Session() as sess:
    print(sess.run(Y, feed_dict = {X: 0.}))
    print(sess.run(Y, feed_dict = {X: 1.}))
    print(sess.run(Y, feed_dict = {X: 2.}))
```

получим ровно то, что и хотели:

```
9.0
16.0
25.0
```

## Переменные

Следующий наш шаг, это научиться решать с помощью TF задачу оптимизации. В этом случае констант и заполнителей будет недостаточно, понадобятся
переменные:

```python
import tensorflow as tf
X = tf.Variable(5.)
with tf.Session() as sess:
    print(sess.run(X))
```

Такой код создаст переменную `X`, и кажется, что всё пройдёт гладко, а в консоль будет выведено `5.0`. Но на самом деле мы получим ошибку на
строке `print(sess.run(X))`. Дело в том, что за начальную инициализацию переменных в рамках сессии отвечает специальная операция, которую надо
вызвать до того, к переменным как обращаться. Поэтому правильный код будет выглядеть так:

 ```python
import tensorflow as tf
X = tf.Variable(5.)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(X))
```

> #### Замечание
> Вызов дополнительной функции кажется странным, но на самом деле в этом есть свой смысл. Например, если мы хотим продолжить тренировку, с некоторой
> контрольной точки, то вместо того, чтобы вызывать процедуру инициализации, мы можем подгрузить начальные значение переменных из файла, в который
> были сохранены промежуточные результаты тренировки и таким образом избежать двойной инициализации.
> 
> Такое себе объяснение, но другого я не знаю.

На самом деле значительно удобнее создавать переменную не напрямую, а через `tf.get_variable`. Выглядит это примерно так:

```python
import tensorflow as tf
X = tf.get_variable('X', [2], initializer = tf.random_uniform_initializer(-1., 1.))
...
```

Здесь мы создали уже вектор `X` с двумя компонентами, который будет инициализирован (когда мы вызовем `tf.global_variables_initializer()`) случайными
величинами с равномерным распределением на отрезке $[-1, 1]$. Различных вариантов инициализаторов в TF достаточно, есть возможность инициализировать
переменную константными значениями, случайными величинами соответстующими различным распределениям, а если хочется чего-то экстрастранного, можно
добавить свой вариант инициализатора. 

Отметим, что если переменная с именем `X` уже создана, то мы при вызове `tf.get_variable` получим ошибку:

```
ValueError: Variable X already exists, disallowed.
```

> #### Замечание
> И здесь есть два варианта. Если мы просто ошибочно пытались создать две переменные с разным предназначением, но одинаковым именем, то имя придётся
> поменять. Однако, если мы создали переменную в одной части кода, то в другой части кода мы можем получить ссылку на вершину с этой переменной в
> графе, используя имя вершины/переменной. Чтобы это сделать надо разобраться с понятием *область видимости переменных* (*variable_scope*), но это мы
> отложим на потом.

## Оптимизация

Для начала попробуем при помощи TF найти минимум выпуклой функции одной переменной. Искать минимум мы будем методом *градиентного спуска*, а значит нам
понадобится `tf.train.GradientDescentOptimizer` и его метод `minimize`. При помощи метода `minimize` мы получим ссылку на операцию в графе вычислений.
Далее мы создадим сессию и внутри неё будем эту операцию раз за разом "вычислять" и таким образом осуществлять тот самый градиентный спуск. При каждом
вызове операция вычислит текущее значение `Y` при текущем значении `X`, подсчитает градиент `Y` по `X` и подвинет `X` в направлении обратном
направлению градиента. 

Операцию можно повторять либо фиксированное число шагов, либо пока мы не решим, что получили минимум достаточно точно. Пример кода, который делает
`50` итераций градиентного спуска для функции $y = x^2 - 2x + 1$:

```python
import tensorflow as tf

learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(0.)
Y = X * X - 2. * X + 1.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50):
        sess.run(train_op)
        if ((step + 1) % 10 == 0):
            x, y = sess.run([X, Y])
            print("{}. x = {:.9f}, y = {:.9f}".format(step + 1, x, y))
```

В результате получим что-то вроде:

```
10. x = 0.892625809, y = 0.011529207
20. x = 0.988470733, y = 0.000132918
30. x = 0.998762012, y = 0.000001550
40. x = 0.999867082, y = 0.000000000
50. x = 0.999985754, y = 0.000000000
```

Результат отличный, минимум нашли, `x` правда определили не точно `1.0`, но мы пока не будем этим заморачиваться.

Можно избавиться от подсчёта итераций, переложив эту задачу на TF:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(0.)
Y = X * X - 2. * X + 1.

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 50):
        _, step = sess.run([train_op, global_step])
        if (step % 10 == 0):
            x, y = sess.run([X, Y])
            print("{}. x = {:.9f}, y = {:.9f}".format(step, x, y))
```

Мы добавили переменную `global_step`. `trainable=False` сообщает TF, чтобы он не помещал новую переменную в список тех, которые надо "тренировать",
т.е. менять при оптимизации. Затем мы передаем тензор `global_step` в метод `minimize` оптимизатора и теперь при каждом запуске операции `train_op`
в нашей сессии, значение в переменной `global_step` будет увеличиваться на единицу. Плюс такого подхода в том, что если мы сохраним некоторую
контрольную точку тренировки, то значение `global_step` сохранится вместе со значения остальных переменных и при загрузке будет поднято из файла.

Ещё у нас в коде есть константа `learning_rate`, которая отвечает за скорость обучения (*learning rate*). В данном случае постоянная скорость
обучения вполне оправдана. Однако, при тренировке, например, нейронных сетей, принято эту скорость обучения менять. Один из стандартных подходов -
это уменьшение скорости обучения со временем (есть и более модные способы). В TF предусмотрена для этого стандартная возможность:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1, global_step, 25, 0.5, True)
...
```

В данном случае начальный `learning_rate` равный `0.1` будет каждые `25` шагов уменьшаться вдвое (умножаться на `0.5`). 

Если в `tf.train.exponential_decay(..., True)` заменить последний параметр на `False`, то вместо дискретного уменьшения, каждые 25 шагов получим
плавное на каждом шаге.  На самом деле вариантов изменения скорости обучения со временем, реализованных в TF достаточно много и разных, а главное ничто
не мешает реализовать свой. 

### Методы оптимизации

С разнообразными методами оптимизации в TF дело обстоит прекрасно и совсем не обязательно использовать обычный градиентный спуск. Например,
практически все методы, которые я разбирал [здесь]({% post_url 2018-03-25-optimization_grad_desc %}) реализованы в TF.

Попробуем найти минимум функции Матьяса:

$$f(x,y) = 0.26 (x^2 + y^2) - 0.48 xy$$

эта функция используется для тестирования алгоритмов поиска оптимума функций. График можно посмотреть, например, 
[на википедии](https://commons.wikimedia.org/wiki/File:Matyas_function.pdf?uselang=ru_). Известно что на множестве $[-10, 10] \times [-10, 10]$
функция Матьяса имеет единственный минимум $f(0, 0) = 0$

Вначале попробуем, аналогично тому как делали выше, обычный градиентный спуск:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = 0.26 * (X * X + Y * Y) - 0.48 * X * Y
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Z, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 500):
        _, step = sess.run([train_op, global_step])
        if (step % 50 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
```

Результат:

```
50. x = -8.181640625, y = -8.183255196, z = 2.678096771
100. x = -6.679733753, y = -6.681998730, z = 1.785360336
150. x = -5.454568386, y = -5.457299232, z = 1.190690994
200. x = -4.464282990, y = -4.465157509, z = 0.797348976
250. x = -3.653036118, y = -3.653432369, z = 0.533844948
300. x = -2.987657547, y = -2.989088535, z = 0.357215405
350. x = -2.444416523, y = -2.445078611, z = 0.239071369
400. x = -1.995408177, y = -1.998004436, z = 0.159475207
450. x = -1.632725000, y = -1.633268237, z = 0.106667161
500. x = -1.334816456, y = -1.335494161, z = 0.071305633
```

Очевидно, что направление выбрано правильно, но добраться до минимум за 500 итераций так и не удалось. Вариантов два, либо пытаться увеличивать
параметр скорости обучения, либо воспользоваться более продвинутым методом оптимизации. График функции мы видели - минимум явно в "овраге", а значит
есть смысл попробовать [метод моментов]({% post_url 2018-03-25-optimization_grad_desc %}#моменты). Немного поменяем код:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)
momentum = tf.constant(0.9, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = 0.26 * (X * X + Y * Y) - 0.48 * X * Y
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(Z, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 500):
        _, step = sess.run([train_op, global_step])
        if (step % 50 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
```

Результат резко улучшился, как и предполагалось:

```
50. x = -0.560837388, y = -0.584768414, z = 0.013267308
100. x =  0.085999884, y =  0.087918639, z = 0.000303397
150. x = -0.001019136, y = -0.001171413, z = 0.000000054
200. x = -0.000417049, y = -0.000408922, z = 0.000000007
250. x =  0.000024892, y =  0.000024595, z = 0.000000000
300. x =  0.000000999, y =  0.000000996, z = 0.000000000
350. x = -0.000000180, y = -0.000000179, z = 0.000000000
400. x =  0.000000004, y =  0.000000004, z = 0.000000000
450. x =  0.000000001, y =  0.000000001, z = 0.000000000
500. x = -0.000000000, y = -0.000000000, z = 0.000000000
```

### Оптимизация по конкретным переменным

Бывает, что у функции многих переменных необходимо найти минимум (максимум) по какому-то подмножеству переменных, при фиксированных остальных.
Например, в крайне популярном сейчас *GAN* подходе используются две составляющии: *генератор* и *дискриминатор*. Генератор и дискриминатор задаются
при помощи нейронных сетей. Тренировка же состоит в оптимизации штрафных функций в которые входят и дискриминатор и генератор, но на каждом шаге
оптимизируются веса либо только дискриминатора, либо только генератора.

В TF для операции оптимизации можно определить какой набор переменных эта операция может менять когда ищет минимум функции. Рассмотрим вот такой
пример:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = X * X + Y * Y
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op_x = optimizer.minimize(Z, global_step=global_step, var_list=[X])
train_op_y = optimizer.minimize(Z, global_step=global_step, var_list=[Y])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Optimize by X:")
    step = 0
    while (step < 50):
        _, step = sess.run([train_op_x, global_step])
        if (step % 10 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
    print("")
    print("Optimize by Y:")
    while (step < 100):
        _, step = sess.run([train_op_y, global_step])
        if (step % 10 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
```

В данном случае, у функции $f(x, y) = x^2 + y^2$ вначале `50` итераций ищется минимум по `X` при фиксированном `Y`, а затем еще `50` итераций ищется
минимум по `Y` при фиксированном `X`. Чтобы обеспечить такое поведение мы завели две операции: `train_op_x` в метод создания которой мы передали список
переменных `var_list` состоящий из одного элемента `X` и `train_op_y` где в переданом списке был только `Y`.

Результат работы данного кода будет таким:

```
Optimize by X:
10. x = -1.073741674, y = -10.000000000, z = 101.152923584
20. x = -0.115292147, y = -10.000000000, z = 100.013290405
30. x = -0.012379400, y = -10.000000000, z = 100.000152588
40. x = -0.001329228, y = -10.000000000, z = 100.000000000
50. x = -0.000142725, y = -10.000000000, z = 100.000000000

Optimize by Y:
60. x = -0.000142725, y = -1.073741674, z = 1.152921200
70. x = -0.000142725, y = -0.115292147, z = 0.013292300
80. x = -0.000142725, y = -0.012379400, z = 0.000153270
90. x = -0.000142725, y = -0.001329228, z = 0.000001787
100. x = -0.000142725, y = -0.000142725, z = 0.000000041
```

Видно, что как и ожидалось, в начале меняется `X` при фиксированном `Y`, а потом наоборот, `Y` сдвигается, а `X` остаётся неизменным.

## Сохранение и восстановление состояния сессии

TF позволяет сохранять значение переменных из сессии в файл, а затем восстанавливать в другой сессии. Для операций сохранения и загрузки используется
класс `tf.train.Saver`. Например, разобьём код из предыдущего параграфа на две части. В первой части:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = X * X + Y * Y
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Z, global_step=global_step, var_list=[X])

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Optimize by X:")
    step = 0
    while (step < 50):
        _, step = sess.run([train_op, global_step])
        if (step % 10 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
    print("")
    saver.save(sess, './model.ckpt')
```

Мы проходим `50` итераций оптимизирующих функцию по переменной `X` и сохраняем текущее состояние в файл `./model.ckpt`. На экране, всё тоже, что мы
наблюдали ранее:

```
Optimize by X:
10. x = -1.073741674, y = -10.000000000, z = 101.152923584
20. x = -0.115292147, y = -10.000000000, z = 100.013290405
30. x = -0.012379400, y = -10.000000000, z = 100.000152588
40. x = -0.001329228, y = -10.000000000, z = 100.000000000
50. x = -0.000142725, y = -10.000000000, z = 100.000000000
```

Во второй части:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = X * X + Y * Y
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Z, global_step=global_step, var_list=[Y])

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    print("Variables restored:")
    step, x, y, z = sess.run([global_step, X, Y, Z])
    print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
    print("Optimize by Y:")
    while (step < 100):
        _, step = sess.run([train_op, global_step])
        if (step % 10 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
```

восстанавливаем значения переменных из файла `./model.ckpt` и продолжаем оптимизировать функцию по `Y`. Результат, не отличается от того, что был в
предыдущем параграфе:

```
Variables restored
50. x = -0.000142725, y = -10.000000000, z = 100.000000000
Optimize by Y:
60. x = -0.000142725, y = -1.073741674, z = 1.152921200
70. x = -0.000142725, y = -0.115292147, z = 0.013292300
80. x = -0.000142725, y = -0.012379400, z = 0.000153270
90. x = -0.000142725, y = -0.001329228, z = 0.000001787
100. x = -0.000142725, y = -0.000142725, z = 0.000000041
```

Отметим, что совершенно не обязательно восстанавливать значения всех переменных. В нашем примере можно ограничиться восстановлением только `X` и `Y`,
а номер итерации обнулить:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(-10.)
Y = tf.Variable(-10.)
Z = X * X + Y * Y
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Z, global_step=global_step, var_list=[Y])

saver = tf.train.Saver([X, Y])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model.ckpt')
    print("Variables restored:")
    step, x, y, z = sess.run([global_step, X, Y, Z])
    print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
    print("Optimize by Y:")
    while (step < 100):
        _, step = sess.run([train_op, global_step])
        if (step % 10 == 0):
            x, y, z = sess.run([X, Y, Z])
            print("{}. x = {:.9f}, y = {:.9f}, z = {:.9f}".format(step, x, y, z))
```

Результат при этом несколько изменится:

```
Variables restored:
0. x = -10.000000000, y = -10.000000000, z = 200.000000000
Optimize by Y:
10. x = -10.000000000, y = -1.073741674, z = 101.152923584
20. x = -10.000000000, y = -0.115292147, z = 100.013290405
30. x = -10.000000000, y = -0.012379400, z = 100.000152588
40. x = -10.000000000, y = -0.001329228, z = 100.000000000
50. x = -10.000000000, y = -0.000142725, z = 100.000000000
60. x = -10.000000000, y = -0.000015325, z = 100.000000000
70. x = -10.000000000, y = -0.000001646, z = 100.000000000
80. x = -10.000000000, y = -0.000000177, z = 100.000000000
90. x = -10.000000000, y = -0.000000019, z = 100.000000000
100. x = -10.000000000, y = -0.000000002, z = 100.000000000
```

> #### Замечание
> Кстати если поменять местами `saver.restore(sess, './model.ckpt')` и `sess.run(tf.global_variables_initializer())`, то всё восстановленное, 
> переинициализируется заново и вернётся к начальным значениям (`X = -10.` и `Y = -10.`).