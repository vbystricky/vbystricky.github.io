---
layout: post
title: Tensorflow (часть вторая)
date: 2019-05-05
category: tensorflow
tags: [tensorflow]
use_math: true
---

Продолжаем разбираться с Tensorflow. Будем смотреть на вспомогательные операции, которые позволяют усложнить граф вычислений таким образом, чтобы поток
данных мог идти по разным путям, в зависимости от содержимого этих данных. Но начнем мы с операции, которая позволит нам логировать сам процесс 
выполнения вычислений в графе.

<!--more-->

## Операция печати

Иногда не плохо было бы иметь возможность вывести что-то на печать непосредственно из графа вычислений, чтобы не выдёргивать значения тензоров из
сессии наружу. Например, так удобнее логировать процесс выполнения или вставлять отладочные сообщения. В TF для добавления в граф операции печати
предусмотрена функция `tf.print`. Можно, например, переписать код оптимизации функции следующим образом:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(0.)
Y = X * X - 2. * X + 1.

print_str = tf.strings.format("{}. x = {}, y = {}", (global_step, X, Y))
print_op = tf.print(print_str)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(Y, global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 50):
        _, step = sess.run([train_op, global_step])
        if (step % 10 == 0):
            sess.run(print_op)
```

Результат будет, практически таким же как и раньше, только с несколько более убогим форматированием:

```
10. x = 0.892625809, y = 0.0115292072
20. x = 0.988470733, y = 0.000132918358
30. x = 0.998762, y = 1.54972076e-06
40. x = 0.999867082, y = 0
50. x = 0.999985754, y = 0
```

## Контроль зависимостей

Когда мы в рамках текущей сессии пытаемся получить значение какого-то тензора, то TF пробегает по всем рёбрам входящим в вершину графа от которой
мы хотим получить тензор, и вычисляет значения в вершинах присоединённых по этим ребрам, а чтобы вычислить их пробегает по рёбрам в них входящим и
так рекурсивно пока не дойдет до вершин без входных рёбер, например, констант, переменных или заполнителей (для последних TF пытается отыскать
значения в переданном `feed_dict`). Однако, иногда надо при выполнении операции в одной вершине, выполнить операцию в другой, никак не связанной с
первой. Например, в примере из предыдущего параграфа было бы удобно, не вызывать операцию печати самому, а заставить TF это делать при каждом вызове
операции оптимизации. В этом нам поможет контроль зависимостей: `tf.control_dependencies`. Эта функция создаёт контекст внутри которого все вновь
созданные операции гарантированно выполнятся позднее тех операций список которых мы передали в функцию в качестве параметра. 

Или иначе говоря вызов любой операции, созданной внутри контекста, дёрнет вначале все операции из списка переданного в `tf.control_dependencies`.
Поправим код нашего примера:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(0.)
Y = X * X - 2. * X + 1.

print_str = tf.strings.format("{}. x = {}, y = {}", (global_step, X, Y))
print_op = tf.print(print_str)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
with tf.control_dependencies([print_op]):
    train_op = optimizer.minimize(Y, global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 50):
        _, step = sess.run([train_op, global_step])
    sess.run(print_op)
```

Результат будет следующим:

```
0. x = 0, y = 1
1. x = 0.2, y = 0.64
2. x = 0.36, y = 0.40959996
3. x = 0.488000035, y = 0.26214397
4. x = 0.59040004, y = 0.167772114
5. x = 0.67232, y = 0.107374191
6. x = 0.73785603, y = 0.0687194467
7. x = 0.790284812, y = 0.0439804792
8. x = 0.832227826, y = 0.0281475186
9. x = 0.865782261, y = 0.018014431
10. x = 0.892625809, y = 0.0115292072
...
```

Стало значительно удобнее, но печатать на каждом шаге не очень хорошо, хотелось бы вернуться к варианту с выводом каждые $N$ шагов. Для этого нам
понадобятся условные операции.

## Условные операции

В TF можно добавлять в граф вершины, которые выполняют роль условного оператора т.е. в зависимости от того, что пришло на вход данной вершины, внутри
будет выполнена та или другая операция. За добавление такой вершины отвечает функция `tf.cond`.

Допустим мы решили подсчитывать при помощи TF корень квадратный из числа подаваемого на вход. Можно написать простой код вроде:

```python
import tensorflow as tf

X = tf.placeholder(dtype = tf.float32, shape = (), name = "X")
Y = tf.sqrt(X)
with tf.Session() as sess:
    print(sess.run(Y, feed_dict = {X : 4}))
    print(sess.run(Y, feed_dict = {X : 9}))
    print(sess.run(Y, feed_dict = {X : -16}))
```

который вполне нормально отработает и выдаст:

```
2.0
3.0
nan
```

Однако, допустим мы хотим, чтобы для отрицательных чисел, мы получали `0` вместо `nan`. Чтобы удовлетворить наше желание, воспользуемся `tf.cond`:

```python
import tensorflow as tf

X = tf.placeholder(dtype = tf.float32, shape = (), name = "X")
Y = tf.cond(tf.greater_equal(X, tf.constant(0.)), 
            lambda: tf.sqrt(X),
            lambda: tf.constant(0.))
with tf.Session() as sess:
    print(sess.run(Y, feed_dict = {X : 4}))
    print(sess.run(Y, feed_dict = {X : 9}))
    print(sess.run(Y, feed_dict = {X : -16}))
```

Результатом будет:

```
2.0
3.0
0.0
```

С одной стороны всё достаточно просто. Три параметра, первый добавляет вершину, которая умеет сравнить `X` и `0.0` (ноль завернули в константу, хотя
можно было и не заворачивать, питон сделал бы это за нас), и две функции, одна вызывается для случая, когда вершина из первого параметра вернет `true`,
вторая если `false`. Однако, есть и нюансы.

Во-первых, функция для `true` и функция для `false` должны возвращать тензоры содержащие значения одинакового типа. Т.е. например, вот такой код:

```python
...
Y = tf.cond(tf.greater_equal(X, tf.constant(0.)), 
            lambda: tf.sqrt(X),
            lambda: tf.constant(0))
...
```

выдаст ошибку:

```
ValueError: Outputs of true_fn and false_fn must have the same type: float32, int32
```

Потому что функция для `true` вернёт скаляр типа `float32`, а функция для `false` вернёт тоже скаляр, но типа `int32`. Забавно, что при этом ранг и
размерности тензоров для `true` и для `false` могут быть разные, т.е. вот такое изменение:

```python
...
Y = tf.cond(tf.greater_equal(X, tf.constant(0.)), 
            lambda: tf.sqrt(X),
            lambda: tf.constant([0., 0.]))
...
```

Отработает вполне нормально.

Есть и более интересные особенности. Рассмотрим, например, вот такой код:

```python
import tensorflow as tf

pred = tf.constant(False)
a = tf.constant(0)

print_true = tf.print("true_fn\n")
print_false = tf.print("false_fn\n")

def true_fn():
    with tf.control_dependencies([print_true]):
        return tf.identity(a)

def false_fn():
    with tf.control_dependencies([print_false]):
        return tf.identity(a)

cond_op = tf.cond(pred, true_fn, false_fn)
with tf.Session() as sess:
    sess.run(cond_op)
```

Казалось бы интуитивно понятно, что в результате должно быть напечатано `false_fn`. Но в результате работы данного кода в консоль будет выведено:

```
false_fn
true_fn
```

Самое интересное, если мы слегка подправим наш код:

```python
import tensorflow as tf

pred = tf.constant(False)
a = tf.constant(0)

def true_fn():
    with tf.control_dependencies([tf.print("true_fn\n")]):
        return tf.identity(a)

def false_fn():
    with tf.control_dependencies([tf.print("false_fn\n")]):
        return tf.identity(a)

cond_op = tf.cond(pred, true_fn, false_fn)
with tf.Session() as sess:
    sess.run(cond_op)
```

то получим результат, который изначально ожидали: только `false_fn` в консоли. Объясняется это следующим образом. Когда мы создаем операцию, вне
функций `true_fn`, `false_fn`, мы фактически создаём вершину в графе, результат работы этой вершины, должен быть вычислен и передан в `tf.cond` до
проверки условия, т.е. не зависимо от того понадобится ли нам этот результат. Если же мы создаем операцию внутри функций `true_fn` или `false_fn`, то
она будет вызывана, после проверки условия и только по необходимости. Такие интересные особенности функции `tf.cond`, при неаккуратном использовании,
могут привести в лучшем случае к существенному замедлению работы, а в худшем еще и к неправильным результатам.

Вернёмся к нашему коду оптимизирующему функцию. Теперь мы можем поправить его таким образом, чтобы печатать не на каждом шаге:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.constant(0.1, dtype = tf.float32)

X = tf.Variable(0.)
Y = X * X - 2. * X + 1.

print_str = tf.strings.format("{}. x = {}, y = {}", (global_step, X, Y))
print_step = tf.constant(10)
print_op = tf.cond(tf.equal(tf.mod(global_step, print_step), tf.zero()),
                   lambda: tf.print(print_str),
                   lambda: tf.no_op())

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
with tf.control_dependencies([print_op]):
    train_op = optimizer.minimize(Y, global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while (step < 50):
        _, step = sess.run([train_op, global_step])
    sess.run(print_op)
```

В результате получится:

```
0. x = 0, y = 1
10. x = 0.892625809, y = 0.0115292072
20. x = 0.988470733, y = 0.000132918358
30. x = 0.998762, y = 1.54972076e-06
40. x = 0.999867082, y = 0
50. x = 0.999985754, y = 0
```

## Операция ветвления

Функция `tf.cond` добавляет простое ветвление, которое в зависимости от выполнения условия вызывает одну из двух функций. Если условий много и при
выполнении каждого надо вызвать свою функцию, то собирать конструкцию рекурсивно из условных операций сложно и неудобно. Дабы упростить жизнь для такой
ситуации в TF предусмотрена функция `tf.case`. В нее передаются либо список пар: (условие, операция), либо словарь в котором ключ - условие, а значение -
операция.

Реализуем в качестве примера функцию:

$$sign(x) = 
\begin{cases}
-1, & x < 0 \\
 0, & x = 0 \\
+1, & x > 0 
\end{cases}
$$

Например, вот такой код:

```python
import tensorflow as tf

X = tf.placeholder(dtype = tf.float32, shape = (), name = "X")
Y = tf.case([(tf.less(X, tf.constant(0.)), lambda: tf.constant(-1.)),
             (tf.equal(X, tf.constant(0.)), lambda: tf.constant(0.)),
             (tf.greater(X, tf.constant(0.)), lambda: tf.constant(1.))])
with tf.Session() as sess:
    print(sess.run(Y, feed_dict = {X : -4.}))
    print(sess.run(Y, feed_dict = {X : 0.}))
    print(sess.run(Y, feed_dict = {X : 5.}))
```

выдаст:

```
-1.0
0.0
1.0
```

Можно использовать словарь, вместо списка:

```python
...
Y = tf.case({tf.less(X, tf.constant(0.)): lambda: tf.constant(-1.),
             tf.equal(X, tf.constant(0.)): lambda: tf.constant(0.),
             tf.greater(X, tf.constant(0.)): lambda: tf.constant(1.)})
...
```

Результат будет тот же.

Для `tf.case` действуют аналогичные ограничения, что и для функции `tf.cond`, т.е. тип значений внутри тензоров выдаваемых при разных условиях должен
быть один и тот же.

Вообще говоря, не обязательно, чтобы при любых значениях параметров выполнялось одно и только одно условие. Например, испортим наш код следующим
образом:

```python
Y = tf.case([(tf.less_equal(X, tf.constant(0.)), lambda: tf.constant(-1.)),
             (tf.equal(X, tf.constant(0.)), lambda: tf.constant(0.)),
             (tf.greater_equal(X, tf.constant(0.)), lambda: tf.constant(1.))])
```

теперь при `X = 0.` все три условия истинны. В таком случае TF запустит первую операцию, для которой выполнилось условие, т.е. результат будет:

```
-1.0
-1.0
1.0
```

А вот если дополнительно при вызове `tf.case` передать параметр `exclusive` равный `True`:

```python
...
Y = tf.case([(tf.less_equal(X, tf.constant(0.)), lambda: tf.constant(-1.)),
             (tf.equal(X, tf.constant(0.)), lambda: tf.constant(0.)),
             (tf.greater_equal(X, tf.constant(0.)), lambda: tf.constant(1.))],
            exclusive = True)
...
```

то TF проверит выполнение всех условий из списка, и, если выполнено больше одного, вернёт ошибку `InvalidArgumentError` и даже попытается расписать
какие условия были выполнены одновременно.

Так же ошибка возникнет, если не выполнено ни одно из условий, т.е. например, вот такой код:

```python
...
Y = tf.case([(tf.less(X, tf.constant(0.)), lambda: tf.constant(-1.)),
             (tf.greater(X, tf.constant(0.)), lambda: tf.constant(1.))])
...
```

вернёт ошибку при `X = 0.`. . В этом случае можно предусмотреть вызов функции по умолчанию и добавить параметр `default`:

```python
...
Y = tf.case([(tf.less(X, tf.constant(0.)), lambda: tf.constant(-1.)),
             (tf.greater(X, tf.constant(0.)), lambda: tf.constant(1.))],
            default = lambda: tf.constant(0.))
...
```

и мы снова можем вычислять функцию $sign(x)$ в нуле.

## Операция, создающая цикл

Следующая операция, которую мы разберем, позволяет создать в графе вычислений подграф (или вершину), внутри которого выполняются одни и те же операции,
пока истинно некоторое условие. За создание такой операции отвечает `tf.while_loop`. Попробуем, например, с помощью этой функции создать граф для
вычисления суммы всех чисел от `1` до `10`:

```python
import tensorflow as tf

i = tf.constant(1)
N = tf.constant(10)
sum = tf.constant(0)
sum_op = tf.while_loop(lambda i, sum: tf.less_equal(i, N),
                       lambda i, sum: [tf.add(i, 1), tf.add(sum, i)],
                       [i, sum])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(sum_op[1]))
```

В `tf.while_loop` в качестве аргументов передаются две функции. Первая `cond`, в нашем примере это:

```lambda i, sum: tf.less(i, N)```

и вторая `body`:

```lambda i, sum: [tf.add(i, 1), tf.add(sum, i)]```

Функции принимают одни и те же аргументы (например, параметр `sum` в функции `cond` совершенно бесполезен, мы его никак не используем внутри, но
вынуждены предусмотреть его в списке аргументов), аргументы это либо тензор, либо набор тензоров. Функция `cond` должна вернуть тензор ранга ноль,
содержащий значения типа `boolean`. У функции `body` выход должен возвращать тензор или набор тензоров, тех же характеристик, что пришли на вход
(этот выход на следующем шаге отправится на вход функций `cond` и `body`). Фактически операция `tf.while_loop` разворачивается в графе в
последовательный вызов этих двух функций, вызвали `cond` проверили результат, если истина вызвали `body`, если ложь вышли из операции.

Важно, что тензоры передаваемые в качестве параметров функции на протяжении всего цикла сохраняют ранг, размерности и тип. А что делать если по ходу
работы цикла нам надо менять характеристики тензора? Ранг тензора и тип значений менять нельзя, но можно менять размерность. Изменим условия нашей
задачи. Допустим мы хотим получить не сумму чисел от `1` до `10`, а вектор содержащий эти числа. Перепишем код следующим образом:

```python
import tensorflow as tf

i = tf.constant(1)
N = tf.constant(10)
series = tf.constant([], dtype = tf.int32)
series = tf.while_loop( lambda i, series:
                            tf.less_equal(i, N),
                        lambda i, series:
                            [tf.add(i, 1), tf.concat([series, [i]], axis = 0)],
                        [i, series])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(series[1]))
```

И при запуске получим ошибку:

```
ValueError: Input tensor 'Const_2:0' enters the loop with shape (1,), but has shape
(2,) after one iteration. To allow the shape to vary across iterations, use the
`shape_invariants` argument of tf.while_loop to specify a less-specific shape.
```

В которой TF логично замечает, что на вход пришёл тензор ранга `1` и размерности `1`, а в следующую итерацию передан тензор ранга `1` и размерности `2`.
А так же есть подсказка воспользоваться параметром `shape_invariants` функции `tf.while_loop`. Действительно, мы можем указать характеристики тензоров,
которые передаются с шага на шаг цикла. При этом, предусмотрена возможность не фиксировать одну или несколько размерностей тензора. У нас в качестве
параметров цикла используется пара тензоров, первый (счётчик цикла `i`) это тензор нулевого ранга. Второй, в который мы планируем собрать
последовательность чисел, это тензор ранга `1`, с размерностью меняющейся от итерации к итерации. В TF есть специальный тип объектов отвечающий за
*shape* (профиль, т.е. ранг плюс размерности), используем его:

```python
...
i_shape = tf.TensorShape([])
series_shape = tf.TensorShape([None])
...
```

Для `series_shape` мы создаём переменную размерность, поэтому вместо числового значения передаём `None`. Перепишем код, используя при создании цикла
параметр `shape_invariants`:

```python
import tensorflow as tf

i = tf.constant(1)
N = tf.constant(10)
series = tf.constant([], dtype = tf.int32)
var_shape = [tf.TensorShape([]), tf.TensorShape([None])]
series = tf.while_loop( lambda i, series: 
                            tf.less(i, N),
                        lambda i, series:
                            [tf.add(i, 1), tf.concat([series, [i]], axis = 0)],
                        [i, series],
                        shape_invariants = var_shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(series[1]))
```

Результатом работы данного кода будет:

```
[1 2 3 4 5 6 7 8 9]
```

Вернемся к нашему коду оптимизирующему функцию $y = x^2 - 2x +1$. Внесём цикл `while` внутрь графа:

```python
import tensorflow as tf

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.constant(0.1, dtype = tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

dumb_value = tf.constant(0)
def cond(dumb_value):
    return tf.less_equal(global_step, 50)

def body(dumb_value):
    X = tf.Variable(0.)
    Y = X * X - 2. * X + 1.

    print_str = tf.strings.format("{}. x = {}, y = {}", (global_step, X, Y))
    print_step = tf.constant(10)
    print_op = tf.cond(tf.equal(tf.mod(global_step, print_step), tf.constant(0)),
                        lambda: tf.print(print_str),
                        lambda: tf.no_op())
    with tf.control_dependencies([print_op]):
        train_op = optimizer.minimize(Y, global_step=global_step)

    return tf.tuple([dumb_value], control_inputs = [train_op])

while_train_op = tf.while_loop(cond, body, [dumb_value])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(while_train_op)
```

Что надо отметить. Первое, создание всех операций, включая операцию оптимизации, внесено во внутрь функции `body`, если это не сделать, то TF решит,
что поскольку операция создана вне цикла, до его создания, то вызывать её надо тоже вне цикла один раз. Второе, в TF невозможно создать цикл без
переменных, т.е. нельзя передать в качестве `loop_var` ни `None`, ни пустой список `[]`. TF в такой ситуации вернет ошибку:

```ValueError: No loop variables provided```

Поэтому мы заведем `dumb_value`. В принципе, можно использовать эту переменную как счетчик цикла и отказаться от `global_step`, а можно не использовать
совсем, как это сделано в коде выше. Наконец, третье, чтобы запустить `train_op`, мы используем функцию `tf.tuple`. Она возвращает список переданных ей
в качестве первого параметра тензоров, но при этом убеждается, что все тензоры из списка вычислены, т.е. завершены операции отвечающие за их
вычисление, при этом дополнительно принимает на вход список операций (`control_inputs`), которые надо выполнить до того как вернуть список тензоров.
Фактически это в некотором смысле тоже самое, что `tf.control_dependencies`.

Результат работы данного кода будет тот же, что и в случае когда цикл организуется средствами питона, а не вносится в граф вычислений. Однако, в 
реальной жизни внесение основного цикла оптимизации внутрь графа может привести к некоторому количеству побочных эфектов, возникновение которых трудно
будет выявить.

