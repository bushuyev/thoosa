import matplotlib.pyplot as plt
import numpy as np


def conv_1(input, response = []):
    output = [0] * (len(input) + len(response ) + 1)
    for i, i_e in enumerate(input):
        for r, r_e in enumerate(response):
            output[i+r] = output[i+r] + i_e * r_e

    return output


def superb(y_1, y_2):
    from operator import add

    return list( map(add, y_1, y_2) )


x = range(0,1000)

sin_y_5Hz = [
    *[ np.sin( 2. * np.pi * x / 200.) for x in range(0, 1000)],
]

sin_y_10Hz = [
    *[ np.sin( 2. * np.pi * x / 100.) for x in range(0, 1000)],
]

sin_y_20Hz = [
    *[ np.sin( 2. * np.pi * x / 50.) for x in range(0, 1000)],
]


lin_y_1 = [
        *[0]*20,
        *[x * 2 for x in range(0, 40)],
        *[80]*40
    ]

lin_y_2 = [
    *[0]*60,
    *[x * -4 for x in range(0, 20)],
    *[-80]*20
]

sin_y_0 = [
    *[0]*20,
    *[ np.sin(np.pi * x / 10.) for x in range(0, 60)],
    *[0]*20
]


sin_y_1 = [
    *[0]*20,
    *[ np.sin(np.pi * x / 60.) for x in range(0, 60)],
    *[0]*20
]





# y = superb(superb(lin_y_1, lin_y_2), sin_y)
# y =  superb(sin_y_10Hz, sin_y_20Hz)
y =  superb(sin_y_5Hz, superb(sin_y_10Hz, sin_y_20Hz))



# kernel = [0.] * 2 #[-10. if x == 10 else 0 for x in range(0, 30)] #[2.]
# kernel = [ np.sin(np.pi * x / 10.) for x in range(0, 30)] #1.
kernel = sin_y_5Hz[:200] #1.
kernel = [-x for x in kernel]
# kernel[1] = -1.

out = conv_1(y, kernel)

plt.scatter(x, y, 1,'r', label="input")
plt.scatter(range(0, len(kernel)), kernel, 3, 'b', label="kernel")
plt.scatter(range(0, len(out)), out, 1, 'g', label="out")

plt.show()