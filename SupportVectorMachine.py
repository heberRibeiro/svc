import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

# Carregamento base de dados
dadosCredito = pd.read_csv('Credit.csv')

# Identificação dos atributos categóricos (tipo 'Object')
atributosParaEncoder = []
for i in list(dadosCredito.columns):
    if(dadosCredito[i].dtype == 'O'):
        atributosParaEncoder.append(i)
del i

# Remoção do atributo "class" da lista de atributos para o encoder
atributosParaEncoder.remove('class')

# Encoder dos atributos do tipo 'Object' para o modelo      
labelencoder = LabelEncoder()
for i in atributosParaEncoder:
    dadosCredito[i] = labelencoder.fit_transform(
            dadosCredito[i])
del i

# Definição dos atributos previsores e do atributo da classe
previsores = dadosCredito.iloc[:, 0:20]#.values
classe = dadosCredito.iloc[:, 20]#.values

# Formação da base de dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, 
                                                    test_size = 0.3, 
                                                    random_state = 0)

# Treinamento do modelo
modelo_svm = SVC(gamma='auto')
modelo_svm.fit(X_train, y_train)

# Teste do modelo
previsoes = modelo_svm.predict(X_test)
confusao = confusion_matrix(y_test, previsoes)
taxaAcerto = accuracy_score(y_test, previsoes)
taxaErro = 1 - taxaAcerto

# Seleção de atributos
forest = ExtraTreesClassifier(n_estimators=10)
forest.fit(X_train,  y_train)
impAttr = list( forest.feature_importances_)

dict = { 'Atributos': [], 'Grau de Importancia do Atributo (%)': []}

for i in range(len(impAttr)):
    dict['Atributos'].append(X_train.columns[i])
    dict['Grau de Importancia do Atributo (%)'].append( 
            round( impAttr[i] * 100.0, 2 ) )
del i
grauImportanciaAtributos = pd.DataFrame(dict, columns=['Atributos', 'Grau de Importancia do Atributo (%)'])
del dict
grauImportanciaAtributos.sort_values(by='Grau de Importancia do Atributo (%)', 
                                    ascending=False, inplace=True)