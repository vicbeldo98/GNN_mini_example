# This file creates a toy dataset. Generated file is located in data/raw
import csv
import random

users = ['Sara', 'Juan', 'Carlos', 'Eva', 'Laura', 'Anna', 'Elena', 'Patricia', 'Jose']
business = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
with open('./yelp_reviews_toy.csv', 'w', newline='') as csvfile:
    row_writer = csv.writer(csvfile)
    for u in users:
        for b in business:
            n = random.random()
            if (n > 0.8):
                row_writer.writerow([u, b, random.randint(0, 5)])
