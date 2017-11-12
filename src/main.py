#!/usr/bin/env python3

from hopfield_model import HopfieldModel

def main():
	image_paths = ["images/car.csv", 
	               "images/mug.csv", 
	               "images/boat.csv",
	               "images/robot.csv",
	               "images/face.csv"]
	hopfield = HopfieldModel()
	hopfield.import_images(image_paths)
	hopfield.normalise()
	hopfield.calc_weights()
	hopfield.set_v0(3)
	hopfield.add_noise(0.1)
	hopfield.run()


if __name__ == "__main__":
	main()