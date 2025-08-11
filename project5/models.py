from django.db import models

class TrajectoryPreference(models.Model):
    trajectory1 = models.JSONField()   # Stores dict/array as JSON
    trajectory2 = models.JSONField()
    choice = models.CharField(max_length=50)  # Can store 'traj1', 'traj2', etc.
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback ({self.id}) - Choice: {self.choice}"
