from utils import folder_operations as fo

def main():
	
	# Insert here your path that contains the TIFF images
	path = 'TIFF images/'
	fo.create_CLAHE(path)


if __name__ == "__main__":
    main()