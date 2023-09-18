from django.db.models import Q, F
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from .models import UserBankAccount, UserBankAccountCluster, UserBankAccountProduct

@receiver(pre_save, sender=UserBankAccount)
def update_fields(sender, instance, **kwargs):

    if instance.balance:
        print('Classes updated!')
        instance.is_churned = False
        if instance.balance == 0.00:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=9)
            instance.cluster_label = UserBankAccountCluster.objects.get(id=9)
            instance.product = UserBankAccountProduct.objects.get(id=1)
            instance.is_churned = True
        elif instance.balance <= 50000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=10)
            instance.product = UserBankAccountProduct.objects.get(id=1)
            instance.is_churned = True
        elif instance.balance <= 150000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=11)
            instance.product = UserBankAccountProduct.objects.get(id=4)
        elif instance.balance <= 250000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=12)
            instance.product = UserBankAccountProduct.objects.get(id=4)
        elif instance.balance <= 350000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=13)
            instance.product = UserBankAccountProduct.objects.get(id=5)
        elif instance.balance <= 450000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=14)
            instance.product = UserBankAccountProduct.objects.get(id=2)
        elif instance.balance > 450000:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=15)
            instance.product = UserBankAccountProduct.objects.get(id=3)
        else:
            instance.cluster_label = UserBankAccountCluster.objects.get(id=16)
            instance.product = UserBankAccountProduct.objects.get(id=1)

