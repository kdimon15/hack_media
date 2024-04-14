import streamlit as st
import pandas as pd
import numpy as np
import catboost
from utils import read_file, create_features, create_illness_features, create_features_for_shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from golden_features import cols_le
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from catboost import Pool, CatBoostRegressor
import plotly.graph_objects as go
import plotly.express as px
import json
from scipy import stats
from phik.report import plot_correlation_matrix
from phik import report
import plotly.figure_factory as ff
import pickle


cat_fts = []

def reset():
    st.session_state.key += 1

def get_shap_df(cur_df):
    return cur_df


@st.cache_resource(show_spinner=True)
def get_shap_values(cur_df, feature_names):
    explainer = shap.TreeExplainer(model)
    test_pool = Pool(cur_df[feature_names].fillna(0), cur_df[target_col])
    shap_values = explainer.shap_values(test_pool)
    print(shap_values)
    return shap_values, explainer


@st.cache_resource(show_spinner=True)
def get_preds(df):
    max_id = df[df['target_1'].notna()].index.max()
    # print(df.iloc[max_id])
    preds = [df.iloc[max_id][target_col] * cb_model.predict(df[cb_model.feature_names_].iloc[max_id+1]) for cb_model in all_models]
    sample_submission = pd.DataFrame({
    'week': ['04.09.2023', '11.09.2023', '18.09.2023', '25.09.2023', '02.10.2023', '09.10.2023', '16.10.2023', '23.10.2023', '30.10.2023', '06.11.2023', '13.11.2023', '20.11.2023', '27.11.2023', '04.12.2023', '11.12.2023', '18.12.2023', '25.12.2023', '01.01.2024', '08.01.2024', '15.01.2024', '22.01.2024', '29.01.2024', '05.02.2024', '12.02.2024', '19.02.2024', '26.02.2024', '04.03.2024', '11.03.2024', '18.03.2024'],
    'revenue': preds
    })
    return sample_submission

numeric_cols = ['До поставки',
                'НРП',
                'Длительность',
                'Сумма',
                'Количество позиций',
                'Количество',
                'Количество обработчиков 7',
                'Количество обработчиков 15',
                'Количество обработчиков 30',
                'Согласование заказа 1',
                'Согласование заказа 2',
                'Согласование заказа 3',
                'Изменение даты поставки 7',
                'Изменение даты поставки 15',
                'Изменение даты поставки 30',
                'Отмена полного деблокирования заказа на закупку',
                'Изменение позиции заказа на закупку: изменение даты поставки на бумаге',
                'Изменение позиции заказа на закупку: дата поставки',
                'Количество циклов согласования',
                'Количество изменений после согласований']

# def sum_plot_chosen_columns(shap_values, df, columns):
#     # получаем индексы колонок
#     idxs = [df.drop(target_col, axis=1).columns.get_loc(x) for x in columns]
#     assert len(idxs) != 1

#     # получаем график для указанных пользователем колонок
#     return st_shap(shap.summary_plot(shap_values[:, idxs], shap_df.drop(target_col, axis=1)[columns]))


# def plot_failures_over_time(data, column, value):
#     # Группировка данных по месяцам и подсчет срывов поставок
#     monthly_failures = data[data[column]==value].groupby('Месяц1')[target_col].sum()
#     for i in range(1, 13):
#         if i not in monthly_failures.index:
#             monthly_failures[i] = 0

#     fig = px.line(monthly_failures, x=range(1, 13), y=target_col, title=f'Динамика срывов поставок по месяцам со значением {value} колонки {column}')
#     fig.update_traces(mode='markers+lines')
#     fig.update_xaxes(title='Месяц')
#     fig.update_yaxes(title='Количество срывов поставок')
#     return fig


def get_filtered_samples(df, dictionary):
    # получаем из df сэмплы с такими же значениями, как в dict {'Поставщик':1}
    query_str = ' and '.join([f"`{key}` == {val}" for key, val in dictionary.items()])
    return df.query(query_str)

st.set_page_config(layout="wide")
st.title('Прогнозирование продаж 📊')

if 'key' not in st.session_state:
    st.session_state.key = 0
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'clicked1' not in st.session_state:
    st.session_state.clicked1 = False
if 'clicked2' not in st.session_state:
    st.session_state.clicked2 = False



with st.expander("Загрузка файла"):
    st.write('')

    file = st.file_uploader(label='Загрузите файл с данными для анализа', accept_multiple_files=False, type=['xlsx'])
    if file is not None:
        uploaded_df = read_file(file)
        extra_df = uploaded_df.copy(deep=True)
        st.session_state.clicked1 = st.button('Получить предсказания и анализ ', type='primary', use_container_width=True)


# with st.expander("Ручной ввод данных"):
#     st.write('')

#     df = pd.read_csv('analytics/inp_template.csv')
#     edited_df = st.data_editor(df, num_rows='dynamic', hide_index=True, use_container_width=True, key=f'editor_{st.session_state.key}')

#     col1, col2 = st.columns(2)
#     col1.button('Очистить таблицу', on_click=reset, type='secondary', use_container_width=True)
#     st.session_state.clicked2 = col2.button('Получить предсказания и анализ', type='primary', use_container_width=True)

#     if st.session_state.clicked2:
#         uploaded_df = edited_df

with st.expander("Доступ по API"):
    st.write('')

    st.markdown(
        """
            **Шаг 1: Получите доступ к API и авторизация** \n
            Прежде чем начать использовать API, удостоверьтесь, что у вас есть доступ и необходимые авторизационные данные,
            такие как ключ API, токен доступа или логин и пароль. Если это требуется, убедитесь, что вы правильно настроили их в вашем коде.

            **Шаг 2: Импортируйте необходимые библиотеки или модули** \n
            Если ваш язык программирования поддерживает импорт библиотек, убедитесь,
            что вы импортировали соответствующие библиотеки для отправки HTTP-запросов и работы с JSON.
            Например, в Python вы можете использовать библиотеку requests.  
                
            """
                )
    
    st.code("""import requests""", language='python')
    
    st.markdown(
        """
            **Шаг 3: Подготовьте данные в формате JSON** \n
            Создайте JSON-объект, который будет содержать данные, которые вы хотите отправить на сервер.
            
            **Шаг 4: Отправьте HTTP-запрос к API** \n
            Используйте выбранную вами библиотеку для отправки HTTP-запроса к API. Укажите URL конечной точки API и передайте данные в формате JSON.
            Вот пример использования библиотеки requests в Python:
            """
                )
    
    st.code('''
            data = {
                "ключ1": "значение1",
                "ключ2": "значение2"
                }
                
            url = "https://api.severstal-analytics.com/get_preds"  
            headers = {
                "Content-Type": "application/json",
                "Authorization": "ваш_токен_доступа"
            }

            response = requests.post(url, json=data, headers=headers)

            # Проверьте статус-код ответа
            if response.status_code == 200:
                # Обработайте успешный ответ от сервера
                response_data = response.json()
                print(response_data)
            else:
                # Обработайте ошибку, если статус-код не 200
                print(f"Ошибка: {response.status_code}")
                print(response.text)
''', language='python')
    
    st.markdown(
        """
            **Шаг 5: Обработайте ответ от сервера** \n
            После отправки запроса, обработайте ответ от сервера. 
            Проверьте статус-код, чтобы убедиться, что запрос выполнен успешно. Если успешно, извлеките данные из ответа JSON и выполните необходимую обработку.
            """
                )

with st.spinner('Загружаем модель...'):
    all_models = pickle.load(open('models.pkl', 'rb'))
    model = all_models[0]
    target_col = 'продажи'
    used_columns = model.feature_names_ + [target_col]


if st.session_state.clicked1 or st.session_state.clicked2 or st.session_state.clicked:
    st.session_state.clicked = True
    tab1, tab2 = st.tabs(['Анализ прогнозирования продаж', 'Анализ заболеваемости и рекламы'])

    # prediction_df используем для предсказания следующих продаж
    prediction_df = create_features(uploaded_df).reset_index(drop=True)

    with tab1:

        preds = get_preds(prediction_df)

        alternative_df = pd.read_excel(file, header=[4, 5])
        alternative_df.columns = [f'{i[0]}_{i[1]}'.replace('\n', '') for i in alternative_df.columns]
        alternative_df.columns = ['год', 'неделя', 'месяц', 'продажи'] + alternative_df.columns.to_list()[4:]
        st.dataframe(alternative_df, height=200)
        # st.dataframe(preds, height=200)
        st.download_button('Скачать таблицу с предсказаниями', data=preds.to_csv(index=False).encode("utf-8"), type='secondary', use_container_width=True, file_name='predicted_sales.csv')


        model = CatBoostRegressor()
        model.load_model('info_model.cbm')

        df = create_features_for_shap(uploaded_df).reset_index(drop=True)
        df = df[df['продажи'].notna()].dropna()
        shap_df = get_shap_df(df)
        shap_values, explainer = get_shap_values(df, model.feature_names_)
        feature_names = df.drop(target_col, axis=1).columns
        # feature_names = [cols_le[x] for x in feature_names]

        st.markdown("### 📚 Как читать эти графики?")

        col1, col2 = st.columns(2)

        col1.markdown("""* Значения слева от центральной вертикальной линии — это **negative** класс, справа — **positive** \n* Чем толще линия на графике, тем больше таких точек наблюдения. \n* Чем краснее точки на графике, тем выше значения фичи в ней.""")
        col2.markdown("""* График помогает определить, какие признаки оказывают наибольшее влияние на количество продаж и в какую сторону – положительную или отрицательную. \n* Длина столбца - это величина вклада этого фактора. Положительная высота означает, что фактор увеличивает предсказание, а отрицательная - уменьшает. \n* Признаки упорядочены сверху вниз в порядке влияния на предсказание.""")

        # дальше выводим разные графики по полученным shap values (можно указывать height и width аргументами в st_shap)
        col1, col2 = st.columns(2)

        with col1:
            st_shap(shap.summary_plot(shap_values, shap_df.drop(target_col, axis=1).to_numpy(), max_display=10, feature_names=feature_names), height=500)

        with col2:
            st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names), height=500)

        st.write('')
        st.write('')
        st.markdown("""###### Влияние признаков на количество продаж - длина и количество синих стрелок увеличивают возможное количество продаж, красные уменьшают.""")

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:],
                shap_df.drop(target_col, axis=1).iloc[0,:], feature_names=feature_names))

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots()  # создаем объекты figure и axes
            perm_importance = permutation_importance(model, shap_df.drop(target_col, axis=1).to_numpy(), shap_df[target_col], n_repeats=5, random_state=42)
            sorted_idx = perm_importance.importances_mean.argsort()[-10:]
            print(perm_importance)

            try:
                data = np.array(perm_importance.importances_mean[sorted_idx])
                if (data <= 0).any():
                    data += 1e-6
                imp = stats.boxcox(data)[0]
            except:
                imp = np.array(perm_importance.importances_mean[sorted_idx])
            theta = np.array(shap_df.drop(target_col, axis=1).columns)[sorted_idx]

            # print(perm_importance.importances_mean[sorted_idx])
            fig = go.Figure(data=go.Scatterpolar(r=[imp[x] for x in range(len(imp)-1, -1, -1)], theta=[theta[x] for x in range(len(imp)-1, -1, -1)]))
            fig.update_traces(fill='toself')
            fig.update_layout(polar = dict(
                              radialaxis_angle = -45,
                              angularaxis = dict(
                              direction = "clockwise",
                              period = 6)
                              ))

            fig.update_layout(title_text="Топ 10 самых важных признаков.")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('###### График использует метод перестановочной важности (permutation importance) для оценки важности признаков. \n* Чем выше значение важности, тем более важным является признак для модели. \n* Если важность признака близка к нулю или отрицательна, это может указывать на то, что данный признак слабо влияет на модель.')

        with col2:
            st.markdown("""###### График показывает влияние на предсказание в разрезе каждого признака и его значений. \n* **Ось X**: Значения признака. \n* **Ось Y**: Важность данного признака для предсказаний модели. Чем выше значение на оси Y, тем более важным является данный признак для модели.""")

            column_to_plot = st.selectbox('Выберите колонку для построения графика:', model.feature_names_)
            st_shap(shap.dependence_plot(column_to_plot, shap_values, shap_df.drop(target_col, axis=1), feature_names=feature_names), height=400)

        st.divider()

        st.markdown('**Корреляционная матрица** - это таблица, отображающая коэффициенты корреляции между различными переменными или признаками в наборе данных. Каждая ячейка в матрице содержит коэффициент корреляции, который измеряет силу и направление связи между двумя соответствующими признаками. Значения ближе к 1 указывают на сильную положительную корреляцию, около 0 указывают на слабую или отсутствующую корреляцию. Аналитики используют корреляционные матрицы для выявления взаимосвязей между переменными и оценки их влияния друг на друга, что полезно для анализа данных и моделирования.')

        phk_df = shap_df.copy()
        phk_df[target_col] = model.predict(shap_df)
        phik_overview = phk_df.phik_matrix(interval_cols=[i for i in phk_df.columns if i != target_col]).round(3).sort_values(target_col)

        data = phik_overview.values
        columns = [cols_le.get(x, x) for x in phik_overview.columns]
        index = [cols_le.get(x, x) for x in phik_overview.index]

        fig = px.imshow(data, x=columns, y=index)

        fig.update_layout(xaxis_title="", yaxis_title="", height=800)

        fig.update_xaxes(tickfont=dict(size=12), tickangle=45)
        fig.update_yaxes(tickfont=dict(size=12))

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # st.button(f'Скачать анализ в PDF', type='primary', use_container_width=True)

        st.divider()



# history
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

    with tab2: 
        st.divider()

        st.dataframe(alternative_df, height=200)

        st.divider()

        st.markdown('##### Анализ предсказания модели')

        col1, col2 = st.columns(2)

        with col1:
            plt.figure(figsize=(8, 5))
            prev = alternative_df[alternative_df['продажи'].notna()]['продажи'].tolist()[-80:]
            new_preds = preds['revenue'].to_list()
            x_plot = len(prev) + len(new_preds)
            y_plot = prev + new_preds
            plt.plot(list(range(len(prev))), prev, color='blue', label='Предыдущие продажи')
            plt.plot(list(range(len(prev), x_plot)), new_preds, color='red', label='Предсказанные продажи')
            plt.xlabel('Дни')
            plt.ylabel('Продажи')
            plt.title('Прогнозирование продаж')
            plt.legend()
            st.pyplot(plt)

        with col2:
            illness_models = pickle.load(open('models_bolezn.pkl', 'rb'))
            ilness_df = create_illness_features(extra_df).reset_index(drop=True)
            max_id = ilness_df[ilness_df['target_1'].notna()].index.max()
            new_preds = [ilness_df.iloc[max_id]['заболеваемость'] * model.predict(ilness_df[illness_models[0].feature_names_].iloc[max_id+1]) for model in illness_models]
            plt.figure(figsize=(8, 5))
            prev = ilness_df[ilness_df['заболеваемость'].notna()]['заболеваемость'].tolist()

            plt.plot(prev + new_preds)
            plt.plot(prev)
            plt.figure(figsize=(8, 5))
            x_plot = len(prev) + len(new_preds)
            y_plot = prev + new_preds
            plt.plot(list(range(len(prev))), prev, color='blue', label='Предыдущие заболевания')
            plt.plot(list(range(len(prev), x_plot)), new_preds, color='red', label='Предсказанные заболевания')
            plt.scatter([36, 88, 140, 192], [prev[x] for x in [36, 88, 140, 192]], color='red', marker='o', label='Идеальные моменты для покупки рекламы')
            plt.xlabel('Дни')
            plt.ylabel('К-во больных')
            plt.title('Прогнозирование к-во больных')
            plt.legend()
            st.pyplot(plt)


        st.divider()

        st.button('Скачать Исторический Анализ в PDF', type='primary', use_container_width=True)

        st.divider()
