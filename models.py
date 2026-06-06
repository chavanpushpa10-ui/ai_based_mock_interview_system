from django.db import models

# Create your models here.
class Resume_data(models.Model):
    user_first_name = models.CharField(max_length=50)
    user_last_name = models.CharField(max_length=50)
    user_email = models.CharField(max_length=50)
    user_resume = models.FileField()
    cv_prediction = models.CharField(max_length=50)
    resume_score = models.CharField(max_length = 50)
    no_of_pages = models.CharField(max_length = 50)
    user_level = models.CharField(max_length = 50)
    actual_skills = models.CharField(max_length = 900)
    recommended_skills = models.CharField(max_length = 900)
    intro_prediction = models.CharField(max_length=50)
    mock_score = models.CharField(max_length=50)
    date = models.DateField()

    def __str__(self):
        return self.user_first_name + " " + self.user_last_name