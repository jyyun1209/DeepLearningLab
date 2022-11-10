import sys
import pickle

def main(argv):
	filePath = argv[1]
	with open(filePath, "rb") as f:
		file = pickle.load(f)
		print(file)


if __name__ == "__main__":
	main(sys.argv)