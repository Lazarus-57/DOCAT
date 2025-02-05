from django.db import models

# Create your models here.
class Cyberbullying(models.Model):
    tweets=models.TextField()
    cyberbullying=models.CharField(max_length=500)