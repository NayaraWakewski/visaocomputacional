Projeto de Visão Computacional do Curso de Data Science da Infinity School

![texto alt](https://cdn.eadplataforma.app/client/infinityschool/upload/others/e993f6d822d15efdb3585d1b733e4020.png)


# Objetivo

O projeto visa construir uma rede Neural convolucional utilizando a biblioteca YOLO, para detecção de imagens.

## ⚙️ Instalando as Bibliotecas necessárias.

**Importando as Bibliotecas**: Nesta etapa, importamos as bibliotecas necessárias para execução do Projeto. Trabalharemos com: 

**pandas as pd:** Para manipulação e análise de dados tabulares.

**import cv2** Importando a biblioteca OpenCV para processamento de imagens em Python


## 🚀 Começando - PIPELINE.

Nesta primeira parte do projeto, realizamos o inicio de um pipeline de objetos usando o modelo YOLO com OpenCV em Python.
Depois de carregar o modelo, as classes e a imagem de exemplo, você pode prosseguir com a detecção de objetos na imagem de exemplo usando o modelo YOLO.

-Carrega o modelo YOLO (You Only Look Once) com os arquivos de configuração 'yolov3.cfg' e os pesos 'yolov3.weights' usando a função cv2.dnn.readNetFromDarknet.

-Carrega as classes reconhecidas pelo modelo YOLO a partir do arquivo 'coco.names' e as armazena em uma lista chamada classes.

-Define cores aleatórias para cada classe e as armazena em uma matriz colors usando np.random.uniform.

-Carrega uma imagem de exemplo chamada 'exemplo.jpg' usando a função cv2.imread.




## 🚀 CRIANDO O BLOB.

Um "blob" é essencialmente uma matriz multidimensional que contém os valores de pixel normalizados da imagem, juntamente com informações adicionais, como dimensões, canais e outras configurações específicas da rede neural.

```python
# Obter as dimensões da imagem
height, width, _ = image.shape

# Construir um blob a partir da imagem para o modelo YOLO
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
```

## 🚀 PREPARANDO O MODELO YOLO


Neste trecho de código, o seguinte está acontecendo:

As camadas de saída do modelo YOLO são definidas. Originalmente, parece que havia uma linha comentada #output_layers_names = net.getUnconnectedOutLayersNames(), que foi substituída por output_names = net.getUnconnectedOutLayersNames(). Isso cria uma lista output_names contendo os nomes das camadas de saída.

Em seguida, as camadas de saída são obtidas a partir do modelo usando net.getLayer(name) para cada nome de camada em output_names. Isso cria uma lista de objetos de camada do OpenCV.

Os nomes das camadas de saída são extraídos desses objetos de camada e armazenados em output_layers.

O código define o blob de entrada da rede neural usando net.setInput(blob). Presumivelmente, o objeto blob foi criado anteriormente a partir da imagem de exemplo usando a função cv2.dnn.blobFromImage.

O modelo YOLO é executado na imagem de exemplo, passando o blob pela rede neural, usando net.forward(output_layers). Isso produz uma lista de saídas das camadas de saída do modelo.

As listas class_ids, confidences, e boxes são inicializadas para armazenar informações sobre detecções de objetos. Estas listas serão preenchidas com informações sobre as detecções de objetos na imagem após a execução do modelo YOLO.

Este trecho do código prepara o modelo YOLO para a detecção de objetos na imagem de exemplo e configura as estruturas de dados necessárias para armazenar as informações sobre as detecções.


## 🚀 PROCESSANDO AS SAÍDAS

Este trecho de código é responsável pelo processamento das saídas da rede neural YOLO, supressão não máxima (NMS) para eliminar detecções redundantes e desenho dos retângulos delimitadores nas imagens de detecção. Aqui está o que cada parte do código faz:

Processar as saídas da rede neural:

O código itera sobre as saídas da rede neural (outputs), que contêm informações sobre possíveis detecções de objetos na imagem.

Para cada detecção em cada saída, são extraídos os escores de confiança para todas as classes e a classe com a maior pontuação (class_id) é determinada.

Se a confiança (confidence) para a detecção for maior que 0,5, isso indica que a detecção é significativa, e suas informações são extraídas, incluindo coordenadas, largura e altura do retângulo delimitador.

As informações da detecção (classe, confiança e caixa delimitadora) são armazenadas nas listas class_ids, confidences e boxes, respectivamente.

Aplicar a supressão não máxima (NMS):

A supressão não máxima é uma técnica usada para eliminar detecções redundantes. O código utiliza a função cv2.dnn.NMSBoxes para calcular as caixas delimitadoras finais após a aplicação do NMS. As caixas, confianças e um limiar (score_threshold e nms_threshold) são fornecidos como entrada.
Desenhar os retângulos delimitadores e exibir as classes nas detecções:

Para cada caixa delimitadora resultante após a aplicação do NMS, o código desenha um retângulo na imagem original usando cv2.rectangle e exibe a classe e a confiança acima do retângulo usando cv2.putText.
Exibir a imagem com as detecções:

Finalmente, o código exibe a imagem com as detecções na janela "Object Detection" usando cv2.imshow. O usuário pode visualizar as detecções e aguardar até que uma tecla seja pressionada (cv2.waitKey(0)) antes de fechar a janela (cv2.destroyAllWindows()).
Este código realiza a detecção de objetos na imagem de exemplo usando o modelo YOLO e exibe a imagem com os retângulos delimitadores e as classes identificadas.


   
### 🛠️ Ferramentas

> **Para elaboração do Projeto utilizamos as seguintes ferramentas:**

- **Visual Studio Code** - utilizado para organizar as etapas do projeto, e criar os notebooks Jupyter.
- **Github** - Plataforma de hospedagem de código, que utilizamos para subir o projeto.



## ⚙️ Para detecção da imagem direto da Webcam, esse seria o código:


- **Importando Bibliotecas**

```python
#Importando Bibliotecas
import cv2
import numpy as np
# Carregar os arquivos do modelo YOLO (os arquivos que nós baixamos)
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# Carregar as classes - e transformando em uma lista
classes = []
f = open('coco.names', 'r')
lines = f.readlines()
f.close()

for line in lines:
    classes.append(line.strip())
# Configurar cores aleatórias para cada classe
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler o próximo quadro da captura de vídeo
    ret, frame = cap.read()

    # Verificar se a captura de vídeo foi bem-sucedida
    if not ret:
        break

    # Obter as dimensões do quadro
    height, width, _ = frame.shape

    # Construir um blob a partir do quadro para o modelo YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Definir as camadas de saída
    output_names = net.getUnconnectedOutLayersNames()
    output_layers = [net.getLayer(name) for name in output_names]
    output_layers = [layer.name for layer in output_layers]

    # Passar o blob pela rede neural
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Configurar as listas de detecção
    class_ids = []
    confidences = []
    boxes = []

    # Processar as saídas da rede neural
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Obter as coordenadas do objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calcular os cantos do retângulo delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Adicionar as informações à lista de detecção
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Aplicar a supressão não máxima para eliminar detecções redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Desenhar os retângulos delimitadores e exibir as classes nas detecções
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'
        color = colors[class_ids[i]]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibir o quadro com as detecções
    cv2.imshow('Object Detection', frame)

    # Parar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
```


## 🎁 Expressões de gratidão

* Compartilhe com outras pessoas esse projeto 📢;
* Quer saber mais sobre o projeto? Entre em contato para tomarmos um :coffee:;
