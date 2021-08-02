# Meibomian-Glands-Recognition

introduzione

## Preparazione

Per riprodurre i risultati ottenuti è necessario inserire il materiale nelle cartelle opportune, nel quale è presente oltre alle maschere e le immagini delle ghiandole anche i classificatori precalcolati e il dataset utilizzato per generarli. 

## Prerequisiti

Necessario un ambiente di sviluppo che supporta Python3 e si consiglia l'installazione di ThunderSVM, in un ambiente che supporta anche CUDA, per eseguire la classificazione utilizzando la GPU e riutilizzare i classificatori precalcolati.
		
## Code

main.py: permette di creare le matrici di covarianza che verranno successivamente splittate in train e test per generare il dataset.
		 
	- Settare la variabile "isUpperGland" a True se si vuole creare le matrici di covarianza per le ghiandole superiori, False altrimenti.
	
	- Settare la variabile "create_matrix" a True se si vuole confermare la creazione delle matrici di covarianza, False altrimenti.
	
	- Cambiare la variabile name se si vuole modificare il nome del file risultante e non si vuole sovrascrivere i risultati precedenti.

main-single.py: permette di creare la matrice di covarianza di una singola immagine, necessario se si vuole predirre l'area della ghiandola in un'immagine:
	
	- Settare la variabile "isUpperGland" a True se si vuole predirre l'area di un'immagine di una ghiandola superiore, False altrimenti.
	
	- Settare la variabile "create_matrix" a True se si vuole confermare la creazione delle matrice di covarianza, False altrimenti.
	
	- Cambiare la variabile name se si vuole modificare il nome del file risultante e non si vuole sovrascrivere i risultati precedenti.
	
	- Selezionare attraverso la variabile ... l'immagine di cui vogliamo fare la predizione. 
	
train-gpu.py: funzione che permette di salvare il classificatore SVM in grado di riconoscere l'area delle ghiandole a partire dalle matrici di covarianza che formano il dataset di train:

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "datasetNotCreate" a True se ancora non è stato fatto il train_test_split e quindi generato il dataset a partire dalla matrice di covarianza, False altrimenti.
	
	- Settare la variabile "isUpperGland" a True se si vuole fare la classificazione su immagini della ghiandola superiore, False altrimenti.
	
	- Settare la variabile "useBestParam" a True se si vuole fare la classificazione attraverso la funzione GridSearchCV che permette di trovare i paramentri migliori per il classificatore, False altrimenti.
	
	- Cambiare il kernel del SVM se si vuole utilizzare uno diverso ('rbf' predefinito).
	
	- Cambiare il nome del file relativo al classificatore per non sovrascrivere i risultati ottenuti
	
predict.py: funzione che permette di predirre l'area delle ghiandole del dataset di test, genera come output un log contenente le accuracy ottenute.

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "isUpperGland" a True se si vuole fare il predict delle ghiandole superiori, False altrimenti.
	
	- Cambiare il nome del file di log in base alla predizione fatta.
	
predict_refactor.py: funzione che permette predirre l'area delle ghiandole di un'immagine richiesta, genera come output l'immagine predetta e la relativa maschera.

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "isUpperGland" a True se si vuole identificare l'area di un'immagine della ghiandola superiore, False altrimenti.
	
	- Cambiare il nome dell'immagine predetta e della relativa maschera in base alla predizione fatta e dell'immagine utilizzta.

