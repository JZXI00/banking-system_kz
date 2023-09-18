from django.db import models

from .constants import TRANSACTION_TYPE_CHOICES
from accounts.models import UserBankAccount


class Transaction(models.Model):
    account = models.ForeignKey(
        UserBankAccount,
        related_name='transactions',
        on_delete=models.CASCADE,
        verbose_name='Шот'
    )
    amount = models.DecimalField(
        decimal_places=2,
        max_digits=12,
        verbose_name='Сома'
    )
    balance_after_transaction = models.DecimalField(
        decimal_places=2,
        max_digits=12,
        verbose_name = 'Транзакциядан кейінгі баланс'
    )
    transaction_type = models.PositiveSmallIntegerField(
        choices=TRANSACTION_TYPE_CHOICES,
        verbose_name = 'Транзакциядан түрі'
    )
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name = 'Уақыты')

    def __str__(self):
        return str(self.account.account_no)

    class Meta:
        ordering = ['timestamp']
        verbose_name = 'Транзакция'  # Set your desired verbose name
        verbose_name_plural = 'Транзакциялар'