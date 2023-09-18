# from decimal import Decimal
# from sklearn.cluster import KMeans
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
import random
from django.contrib.auth.models import AbstractUser
from django.core.validators import (
    MinValueValidator,
    MaxValueValidator,
)
from django.db import models
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver

from .constants import GENDER_CHOICE
from .managers import UserManager
from faker import Faker

fake = Faker()


class User(AbstractUser):
    username = None
    email = models.EmailField(unique=True, null=False, blank=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email

    @property
    def balance(self):
        if hasattr(self, 'account'):
            return self.account.balance
        return 0
    class Meta:
        verbose_name = 'Қолданушы'  # Set your desired verbose name
        verbose_name_plural = 'Қолданушылар'

class BankAccountType(models.Model):
    name = models.CharField(max_length=128, verbose_name='Атауы')
    maximum_withdrawal_amount = models.DecimalField(
        decimal_places=2,
        max_digits=12,
        verbose_name = 'Лимит'
    )
    annual_interest_rate = models.DecimalField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        decimal_places=2,
        max_digits=5,
        help_text='Пайыз мөлшерлемесі 0 - 100 дейін',
        verbose_name = 'Пайыз мөлшерлемесі'
    )
    interest_calculation_per_year = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(12)],
        help_text='Жылына пайыздар қанша рет есептелу керек',
        verbose_name = 'Жылғы пайыздар'
    )

    def __str__(self):
        return self.name

    def calculate_interest(self, principal):
        p = principal
        r = self.annual_interest_rate
        n = Decimal(self.interest_calculation_per_year)

        # Basic Future Value formula to calculate interest
        interest = (p * (1 + ((r/100) / n))) - p

        return round(interest, 2)
    class Meta:
        verbose_name = 'Шот түрі'  # Set your desired verbose name
        verbose_name_plural = 'Шот түрлері'

class UserBankAccountProduct(models.Model):
    id = models.AutoField(primary_key=True)
    product_code = models.CharField(max_length=100, blank=True, null=True, verbose_name='Өнімнің коды')
    product_name = models.CharField(max_length=100, blank=True, null=True, verbose_name='Өнімнің атауы')
    description = models.TextField(blank=True, null=True, verbose_name='Өнімнің сипаттамасы')
    def __str__(self):
        return str(self.product_name)

    class Meta:
        verbose_name = 'Ұсынылатын өнім'  # Set your desired verbose name
        verbose_name_plural = 'Ұсынылатын өнімдер'

class UserBankAccountCluster(models.Model):
    id = models.AutoField(primary_key=True)
    cluster_code = models.CharField(max_length=100, blank=True, null=True, verbose_name='Класстың коды')
    cluster_name = models.CharField(max_length=100, blank=True, null=True, verbose_name='Класстың атауы')
    description = models.TextField(blank=True, null=True, verbose_name='Класстың сипаттамасы')
    def __str__(self):
        return str(self.cluster_name)
    class Meta:
        verbose_name = 'Шот класы'  # Set your desired verbose name
        verbose_name_plural = 'Шот класстары'

class UserBankAccount(models.Model):
    user = models.OneToOneField(
        User,
        related_name='account',
        on_delete=models.CASCADE,
        verbose_name="Тұтынушы"
    )
    account_type = models.ForeignKey(
        BankAccountType,
        related_name='accounts',
        on_delete=models.CASCADE,
        verbose_name = "Шот түрі"
    )
    account_no = models.PositiveIntegerField(unique=True,  verbose_name = "Шот нөмірі")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICE, verbose_name = "Жынысы")
    birth_date = models.DateField(null=True, blank=True,  verbose_name = "Туылған күні")
    balance = models.DecimalField(
        default=0,
        max_digits=12,
        decimal_places=2,
        verbose_name="Баланс"
    )
    interest_start_date = models.DateField(
        null=True, blank=True,
        help_text=(
            'Пайыз есептеу басталатын ай'
        ),
        editable=False
    )
    initial_deposit_date = models.DateField(null=True, blank=True, editable=False)
    cluster_label = models.ForeignKey(UserBankAccountCluster, on_delete=models.SET_NULL, blank=True, null=True,  verbose_name = "Класс")
    product = models.ForeignKey(UserBankAccountProduct, on_delete=models.SET_NULL, blank=True, null=True,  verbose_name = "Өнім ұсыныстары")
    is_churned = models.BooleanField(default=False, verbose_name = "Шығынға жатады ма?")

    def __str__(self):
        return str(self.account_no)


    def get_interest_calculation_months(self):
        interval = int(
            12 / self.account_type.interest_calculation_per_year
        )
        start = self.interest_start_date.month
        return [i for i in range(start, 13, interval)]

    class Meta:
        verbose_name = 'Тұтынушы шоты'  # Set your desired verbose name
        verbose_name_plural = 'Тұтынушы шоттары'


    # def get_cluster(self):
    #     account_data = UserBankAccount.objects.all().values('gender', 'balance', 'interest_start_date')
    #     account_df = pd.DataFrame(account_data)
    #     account_df['gender'] = account_df['gender'].map({'M': 0, 'F': 1})
    #     account_df = account_df.fillna(account_df.mean())
    #
    #     # Банк шоты деректері бойынша K-means кластерлеуді орындау
    #     kmeans = KMeans(n_clusters=3, random_state=0)
    #     kmeans.fit(account_df)
    #
    #     # Әрбір банк шотына кластер белгілерін тағайындау
    #     labels = kmeans.labels_
    #     for i, label in enumerate(labels):
    #         account = UserBankAccount.objects.get(id=i + 1)
    #         account.cluster_label = label
    #         account.save()
    #
    # def churn_selecton(self):
    #     # Дерекқордан банк шоты деректерін шығарып алу және оны алдын ала өңдеу
    #     account_data = UserBankAccount.objects.all().values('gender', 'balance', 'interest_start_date',
    #                                                         'last_transaction_date', 'is_churned')
    #     account_df = pd.DataFrame(account_data)
    #     account_df['gender'] = account_df['gender'].map({'M': 0, 'F': 1})
    #     account_df = account_df.fillna(account_df.mean())
    #
    #     # Деректерді жаттықтыру және сынақ жиындарына бөлу
    #     X = account_df[['gender', 'balance', 'interest_start_date', 'last_transaction_date']]
    #     y = account_df['is_churned']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    #     # Оқыту деректері бойынша логистикалық регрессия моделін оқыту
    #     lr_model = LogisticRegression()
    #     lr_model.fit(X_train, y_train)
    #
    #     # Тестілеу деректері бойынша модельді бағалау
    #     score = lr_model.score(X_test, y_test)
    #     print(f"Тестілеу деректері бойынша дәлдік ұпайы: {score}")
    #
    #     # Барлық банк шоттары бойынша клиенттер шығынын болжау және дерекқорды жаңарту
    #     for account in UserBankAccount.objects.all():
    #         x = [[account.gender, account.balance, account.interest_start_date, account.last_transaction_date]]
    #         is_churned = lr_model.predict(x)[0]
    #         account.is_churned = is_churned
    #         account.save()
    #
    #
    # def product_recommendation(self, account_df):
    #     # Банктік шоттар арасындағы косинустардың ұқсастық матрицасын есептеу
    #     cosine_sim = cosine_similarity(account_df)
    #
    #     # Кластер белгісіне негізделген әрбір банк шоты үшін өнім ұсыныстарын жасау
    #     for i, account in enumerate(UserBankAccount.objects.all()):
    #         sim_scores = list(enumerate(cosine_sim[i]))
    #         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #         sim_scores = sim_scores[1:4]  # Get top 3 most similar accounts
    #         similar_accounts = [UserBankAccount.objects.get(id=j + 1) for j, _ in sim_scores]
    #         recommended_accounts = UserBankAccount.objects.filter(cluster_label=account.cluster_label).exclude(
    #             id=account.id).intersection(similar_accounts)
    #         account.recommended_accounts.set(recommended_accounts)
    #         account.save()



class UserAddress(models.Model):
    user = models.OneToOneField(
        User,
        related_name='address',
        on_delete=models.CASCADE,
        verbose_name = 'Тұтынушы'
    )
    street_address = models.CharField(max_length=512, verbose_name='Көшесі')
    city = models.CharField(max_length=256, default='Алматы', verbose_name='Атауы')
    postal_code = models.PositiveIntegerField(verbose_name='Индекс')
    country = models.CharField(max_length=256, default='Қазақстан', verbose_name='Елі')

    def __str__(self):
        return self.user.email

    class Meta:
        verbose_name = 'Мекенжай'  # Set your desired verbose name
        verbose_name_plural = 'Мекенжайлар'


def create_fake_user_bank_accounts(num_accounts):
    for _ in range(num_accounts):
        # Create a random user
        username = fake.user_name()
        password = fake.password()
        email = fake.email()
        user = User.objects.create_user(password=password, email=email)

        # Кездейсоқ банктік шот түрін жасау
        account_type = BankAccountType.objects.get(id=2)

        # Банк шоты үшін кездейсоқ мәндерді жасау
        account_no = random.randint(100000, 999999)
        gender = random.choice([choice[0] for choice in GENDER_CHOICE])
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=65)
        balance = random.uniform(0, 700000)
        interest_start_date = fake.date_between(start_date='-1y', end_date='today')
        initial_deposit_date = timezone.now()

        # Кездейсоқ кластер белгісін және өнімді жасау
        cluster_label = UserBankAccountCluster.objects.order_by('?').first()
        product = UserBankAccountProduct.objects.order_by('?').first()

        # Пайдаланушы банк шотын жасау
        UserBankAccount.objects.create(
            user=user,
            account_type=account_type,
            account_no=account_no,
            gender=gender,
            birth_date=birth_date,
            balance=balance,
            interest_start_date=interest_start_date,
            initial_deposit_date=initial_deposit_date,
            cluster_label=cluster_label,
            product=product,
            is_churned=False
        )
        print(_)