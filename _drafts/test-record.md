---
layout: post
title: Первая запись для тестов
date: 2017-01-19
category: dummy
tags: [test]
use_math: true
---

Данная запись создана для теста разметки и прочего. Мы сейчас тут текстика добавим.

> "Когда Наташа вышла из гостиной и побежала, она добежала только до цветочной. В этой комнате она остановилась, прислушиваясь к говору в гостиной и ожидая выхода Бориса. Она уже начинала приходить в нетерпение и, топнув ножкой, сбиралась было заплакать оттого, что он не сейчас шел, когда заслышались не тихие, не быстрые, приличные шаги молодого человека. Наташа быстро бросилась между кадок цветов и спряталась.
> <!--more-->
>
> Борис остановился посреди комнаты, оглянулся, смахнул рукой соринки с рукава мундира и подошел к зеркалу, рассматривая свое красивое лицо. Наташа, притихнув, выглядывала из своей засады, ожидая, что он будет делать. Он постоял несколько времени перед зеркалом, улыбнулся и пошел к выходной двери. Наташа хотела его окликнуть, но потом раздумала.
>
> — Пускай ищет, — сказала она себе. Только что Борис вышел, как из другой двери вышла раскрасневшаяся Соня, сквозь слезы что-то злобно шепчущая. Наташа удержалась от своего первого движения выбежать к ней и осталась в своей засаде, как под шапкой-невидимкой, высматривая, что делалось на свете. Она испытывала особое новое наслаждение. Соня шептала что-то и оглядывалась на дверь гостиной. Из двери вышел Николай."

*&copy; Толстой Л.Н. "Война и мир"*

## Тесты разметки кода в разных языках

### Тест кода вообще

```
int main(int argc, char **argv)
{
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```

### Тест кода C++

```c++
//Very very usefull comment
int main(int argc, char **argv)
{
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```

### Тест кода C

```c
int main(int argc, char **argv)
{
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```

### Тест кода Python

```python
import tensorflow as tf
# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```

### Тест кода Ruby

```ruby
def show
  @widget = Widget(params[:id])
  respond_to do |format|
    format.html # show.html.erb
    format.json { render json: @widget }
  end
end
```


## Тесты математических формул

Прямо вот формулы внутри строки текста $f(x_1, x_2, ..., x_N) = g(x_1) \cdot g(x_2) \cdot ... \cdot g(x_N)$

А потом на отдельной строке:

$$
   |\psi_1\rangle = a|0\rangle + b|1\rangle
$$

Нумеруем формулы:

\begin{equation}
   |\psi_1\rangle = a|0\rangle + b|1\rangle
\end{equation}

\begin{equation}
   P(x | \theta) = f(x, \theta)
\end{equation}

или не нумеруем:

$$
   P(x | \theta) = f(x, \theta)
$$

