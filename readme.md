# Meibomian-Glands-Recognition

Il progetto ha l'obiettivo di riconoscere l'area delle ghiandole superiori e inferiori del meibomio per un insieme di immagini fornite.

## Preparazione

Per riprodurre i risultati ottenuti è necessario inserire il materiale nelle cartelle opportune, nel quale è presente oltre alle maschere e le immagini delle ghiandole anche i classificatori precalcolati e il dataset utilizzato per generarli. 

## Prerequisiti

Necessario un ambiente di sviluppo che supporta Python3 e si consiglia l'installazione di ThunderSVM, in un ambiente che supporta anche CUDA, per eseguire la classificazione utilizzando la GPU e riutilizzare i classificatori precalcolati.
		
## Gabor parameters

Si indica con `+` eventuali parametri modificabili.
Parametri Gabor filter:

	+ num_theta: represents the orientation of the normal to the parallel stripes and, given that the shape is vertical in most cases, we have to use few orientation otherwise it could give a strong response even for lashes (default: 2).
	
	+ gabor_ker_size: that is a Gaussian kernel modulated by a complex harmonic function and it would be bigger to identify a wide area from the image (default: 129).
	
	+ lam:  represents the wavelength of the sinusoidal factor (default: 1.01, 1.015).
	
	+ sigma: is the sigma/standard deviation of the Gaussian envelope (default: gabor_ker_size/5).
	
	+ psi: is the phase offset (default: π/2).
	
	+ gamma: is the spatial aspect ratio and specifies the ellipticity of the support of the Gabor function (default: 1.5).


### Code

`create_mask.py`: permette di selezionare a mano l'area della ghiandola e generare la relativa maschera.
		 
	- Settare la variabile "isUpperGland" a True se si vuole creare le maschere per le ghiandole superiori, False altrimenti.
	
	- Settare la variabile "create_mask_matrix" a True se si vuole confermare la creazione delle matrici contenenti le maschere, False altrimenti.
	
	- Settare la variabile "create_mask_image" a True se si vuole confermare la creazione delle immagini delle maschere, False altrimenti.
	
	+ n_segments: numero di segmenti presenti nell'immagine cliccabile a mano che permette di generare la maschera (default: 180).

`main.py`: permette di creare le matrici di covarianza che verranno successivamente splittate in train e test per generare il dataset.
		 
	- Settare la variabile "isUpperGland" a True se si vuole creare le matrici di covarianza per le ghiandole superiori, False altrimenti.
	
	- Settare la variabile "create_matrix" a True se si vuole confermare la creazione delle matrici di covarianza, False altrimenti.
	
	- Cambiare la variabile name se si vuole modificare il nome del file risultante e non si vuole sovrascrivere i risultati precedenti.
	
	+ tile: dimensione delle celle della griglia (default: 10)
	
	+ balance_dataset: percentuale che indica il bilanciamento di esempi negativi/positivi per evitare un dataset sbilanciato (default: 30%)

`main-single.py`: permette di creare la matrice di covarianza di una singola immagine, necessario se si vuole predirre l'area della ghiandola in un'immagine:
	
	- Settare la variabile "isUpperGland" a True se si vuole predirre l'area di un'immagine di una ghiandola superiore, False altrimenti.
	
	- Settare la variabile "create_matrix" a True se si vuole confermare la creazione delle matrice di covarianza, False altrimenti.
	
	- Cambiare la variabile name se si vuole modificare il nome del file risultante e non si vuole sovrascrivere i risultati precedenti.
	
	- Selezionare attraverso la variabile ... l'immagine di cui vogliamo fare la predizione. 
	
`train-gpu.py`: funzione che permette di salvare il classificatore SVM in grado di riconoscere l'area delle ghiandole a partire dalle matrici di covarianza che formano il dataset di train:

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "datasetNotCreate" a True se ancora non è stato fatto il train_test_split e quindi generato il dataset a partire dalla matrice di covarianza, False altrimenti.
	
	- Settare la variabile "isUpperGland" a True se si vuole fare la classificazione su immagini della ghiandola superiore, False altrimenti.
	
	- Settare la variabile "useBestParam" a True se si vuole fare la classificazione attraverso la funzione GridSearchCV che permette di trovare i paramentri migliori per il classificatore, False altrimenti.
	
	- Cambiare il kernel del SVM se si vuole utilizzare uno diverso ('rbf' predefinito).
	
	- Cambiare il nome del file relativo al classificatore per non sovrascrivere i risultati ottenuti
	
`predict.py`: funzione che permette di predirre l'area delle ghiandole del dataset di test, genera come output un log contenente le accuracy ottenute.

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "isUpperGland" a True se si vuole fare il predict delle ghiandole superiori, False altrimenti.
	
	- Cambiare il nome del file di log in base alla predizione fatta.
	
`predict_refactor.py`: funzione che permette predirre l'area delle ghiandole di un'immagine richiesta, genera come output l'immagine predetta e la relativa maschera.

	- Settare la variabile "selectGPU" in base alla GPU da utilizzare.
	
	- Settare la variabile "isUpperGland" a True se si vuole identificare l'area di un'immagine della ghiandola superiore, False altrimenti.
	
	- Cambiare il nome dell'immagine predetta e della relativa maschera in base alla predizione fatta e dell'immagine utilizzta.

folder *utils* contiene le funzioni:
	
	* `balance_dataset.py`
	