import random
import math


class RNA(object):

    def __init__(self,
                 entradas=3,
                 arquitectura=[2, 1],
                 alpha=0.5,
                 aleatoria=None,
                 permitir_logs=False):

        self.red = []
        self.arquitectura = arquitectura
        self.n_capas = len(arquitectura)
        self.n_entradas = entradas
        self.alpha = alpha

        self.aleatoria = aleatoria
        self.permitir_logs = permitir_logs

    def f_aleatoria(self):
        try:
            if self.aleatoria is None:
                return random.uniform(-1, 1)
            else:
                return self.aleatoria()
        except Exception as exc:
            print('RN: Ha ocurrido un error en la funcion aleatoria')
            raise exc

    def generar(self):
        for c in range(self.n_capas):
            self.red.append([])

            for i in range(self.arquitectura[c]):
                if c == 0:
                    self.red[c].append({'pesos': [self.f_aleatoria() for j in range(self.n_entradas + 1)]})
                else:
                    self.red[c].append({'pesos': [self.f_aleatoria() for j in range(self.arquitectura[c - 1] + 1)]})

    def mult(self, entrada, pesos):
        ret = 0
        for i in range(len(entrada)):
            ret += entrada[i] * pesos[i]
        return ret

    def activacion(self, raw):  # raw = salida en bruto de la neurona
        return 1 / (1 + math.exp(-raw))

    def activacion_der(self, out):  # out = salida de activacion de la neurona
        return out * (1 - out)

    def fw_prop(self, ejemplo):  # propagar hacia adelante
        salida_capa = [1] + ejemplo  # bias

        for capa in self.red:
            buffer = [1]  # bias

            for n in capa:
                salida = self.mult(salida_capa, n['pesos'])
                n['salida'] = self.activacion(salida)
                buffer.append(n['salida'])

            salida_capa = buffer

        return salida_capa[1:]  # resultado de la capa de salida

    def bw_prop(self, deseada):  # retro-propagar la salida
        for c in reversed(range(len(self.red))):

            if c == len(self.red) - 1:  # capa de salida
                for n in range(len(self.red[c])):
                    error = (deseada[n] - self.red[c][n]['salida'])
                    self.red[c][n]['delta'] = error * self.activacion_der(self.red[c][n]['salida'])

            else:  # capas ocultas
                for n in range(len(self.red[c])):
                    error_buffer = 0.0

                    for neurona in self.red[c + 1]:  # sumatorio de pesos*deltas anteriores
                        error_buffer += (neurona['pesos'][n + 1] * neurona['delta'])  # bias

                    self.red[c][n]['delta'] = error_buffer * self.activacion_der(self.red[c][n]['salida'])

    def corregir_pesos(self, ejemplo_entrada):  # calculo de incrementos de peso
        for c in range(len(self.red)):

            if c == 0:
                salida_anterior = [1] + ejemplo_entrada
            else:
                salida_anterior = [1] + [n['salida'] for n in self.red[c - 1]]

            for neurona in self.red[c]:
                for i in range(len(salida_anterior)):
                    neurona['pesos'][i] += self.alpha * neurona['delta'] * salida_anterior[i]

    def RMSE(self, buffer, n_ejemplos):
        return math.sqrt(buffer / n_ejemplos)

    def GRMSE(self, buffers, n_salidas):
        return sum(buffers) / n_salidas

    def entrenar(self, ejemplos_entrada, ejemplos_salida, epochs):
        for epoch in range(epochs):

            informe = self.single_epoch(ejemplos_entrada, ejemplos_salida)
            informe['numero'] = epoch

            if self.permitir_logs:
                print('--> Epoch numero {}:'.format(informe['numero']))
                for i in range(len(informe['rmse'])):
                    print('\tRMSE salida {}: {}'.format(i, informe['rmse'][i]))

                print('\tGRMSE de la red: {}'.format(informe['grmse']))

    def single_epoch(self, ejemplos_entrada, ejemplos_salida):
        informe_epoch = {'deseado': [], 'obtenido': [], 'rmse': [], 'grmse': None}

        error_buffer = [0 for i in range(len(ejemplos_salida[0]))]  # one buffer per net output

        for i in range(len(ejemplos_entrada)):

            salida = self.fw_prop(ejemplos_entrada[i])

            self.bw_prop(ejemplos_salida[i])

            self.corregir_pesos(ejemplos_entrada[i])

            # sumatorio de errores por salida
            for j in range(len(salida)):
                error_buffer[j] += sum([(ejemplos_salida[i][j] - salida[j]) ** 2])

            informe_epoch['deseado'].append(ejemplos_salida[i])
            informe_epoch['obtenido'].append(salida)

        # fin de epoch

        for k in range(len(error_buffer)):  # calculo de errores
            informe_epoch['rmse'].append(self.RMSE(error_buffer[k], len(ejemplos_entrada)))

        informe_epoch['grmse'] = self.GRMSE(informe_epoch['rmse'], len(informe_epoch['rmse']))

        return informe_epoch


if __name__ == '__main__':

    red = RNA(entradas=2,
              alpha=0.5,
              arquitectura=[2, 1],
              permitir_logs=True)

    red.generar()

    XOR_in_samples = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    XOR_out_samples = [
        [0],
        [1],
        [1],
        [0]
    ]

    red.entrenar(XOR_in_samples, XOR_out_samples, 100000)

    for e in XOR_in_samples:

        print('Input: {}'.format(e))
        res = red.fw_prop(e)  # propagar hacia delante la entrada
        print('Output: {}'.format(res))
        print('\n')

    input('Press Enter to continue')

    # Ejemplo usando clases de salida

    red = RNA(entradas=2,
              alpha=0.7,
              arquitectura=[2, 2],  # 2 neuronas en la capa de salida
              permitir_logs=True)

    red.generar()

    XOR_in_samples = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # La primera neurona representa la clase (1) y la segunda a (0)
    XOR_out_class_samples = [
        # [(1), (0)]
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ]

    red.entrenar(XOR_in_samples, XOR_out_class_samples, 100000)

    for e in XOR_in_samples:

        print('Entrada: {}'.format(e))
        true_val, false_val = red.fw_prop(e)

        if true_val < 0.01 and false_val > 0.99:
            res = 0
        else:
            res = 1

        print('Salida: {}'.format(res))
        print('\n')
