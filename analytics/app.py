import streamlit as st
import pandas as pd
import numpy as np
import catboost
from utils import read_file, create_features
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
def get_shap_values(cur_df):
    explainer = shap.TreeExplainer(model)
    test_pool = Pool(cur_df.drop(target_col, axis=1), cur_df[target_col])
    shap_values = explainer.shap_values(test_pool)
    print(shap_values)
    return shap_values, explainer


@st.cache_resource(show_spinner=True)
def get_preds(df):
    max_id = df[df['target_1'].notna()].index.max()
    # print(df.iloc[max_id])
    preds = [df.iloc[max_id][target_col] * cb_model.predict(df[all_models[0].feature_names_].iloc[max_id+1]) for cb_model in all_models]
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

# функция, чтобы выводить в summary_plot только выбранные пользователем колонки
def sum_plot_chosen_columns(shap_values, df, columns):
    # получаем индексы колонок
    idxs = [df.drop(target_col, axis=1).columns.get_loc(x) for x in columns]
    assert len(idxs) != 1

    # получаем график для указанных пользователем колонок
    return st_shap(shap.summary_plot(shap_values[:, idxs], shap_df.drop(target_col, axis=1)[columns]))


def plot_failures_over_time(data, column, value):
    # Группировка данных по месяцам и подсчет срывов поставок
    monthly_failures = data[data[column]==value].groupby('Месяц1')[target_col].sum()
    for i in range(1, 13):
        if i not in monthly_failures.index:
            monthly_failures[i] = 0

    fig = px.line(monthly_failures, x=range(1, 13), y=target_col, title=f'Динамика срывов поставок по месяцам со значением {value} колонки {column}')
    fig.update_traces(mode='markers+lines')
    fig.update_xaxes(title='Месяц')
    fig.update_yaxes(title='Количество срывов поставок')
    return fig


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

# with st.spinner('Загружаем данные...'):
#     df = load_data()
#     supp_stat = load_supp_stat()


if st.session_state.clicked1 or st.session_state.clicked2 or st.session_state.clicked:
    st.session_state.clicked = True
    tab1, tab2 = st.tabs(['Анализ прогнозирования продаж', 'Анализ заболеваемости и рекламы'])

    # prediction_df используем для предсказания следующих продаж
    prediction_df = create_features(uploaded_df).reset_index(drop=True)

    with tab1:

        preds = get_preds(prediction_df)

        st.dataframe(pd.read_excel(file, header=[4, 5]), height=200)

        # st.divider()

        # st.markdown('###### Общий анализ')

        # col1, col2, col3, col4 = st.columns(4)

        # y_count = uploaded_df['Предсказание'].value_counts()

        # col1.metric('Количество просрочек', y_count['Просрочка'] if len(y_count.index) == 2 else 0)
        # col2.metric('Количество своевременных поставок', y_count['В срок'] if len(y_count.index) == 2 else 0)
        # col3.metric('Средний риск', str(round(uploaded_df['Риск'].mean(), 2))+'%')
        # col4.metric('Средний рейтинг поставщика', str(round(np.mean([int(supp_stat[str(round(int(x['Поставщик'])))][0]) if str(round(int(x['Поставщик']))) in list(supp_stat.keys()) else 0 for i, x in uploaded_df.iterrows()]), 2))+'⭐')

        # st.divider()

        # sample = uploaded_df.iloc[int(option)]

        # st.markdown(f'###### Анализ для {option} семпла')

        # col1, col2, col3, col4 = st.columns(4)

        # col1.metric('Предсказание модели', '0' if sample['Предсказание'] == 'В срок' else '1', delta = 'Товар поступит в срок!' if sample['Предсказание'] == 'В срок' else 'Товар задержится', delta_color = 'normal' if sample['Предсказание'] == 'В срок' else 'inverse')
        # col2.metric('Риск', str(sample['Риск'])+'%', delta = 'Низкий' if sample['Риск'] < 30 else 'Высокий', delta_color = 'normal' if sample['Риск'] < 30 else 'inverse')
        # col3.metric('Уверенность модели', str(sample['Уверенность'])+'%', delta = 'Высокая' if sample['Уверенность'] > 60 else 'Слабая', delta_color = 'normal' if sample['Уверенность'] > 60 else 'inverse')
        # rv = round(int(sample['Поставщик']))
        # if str(rv) in list(supp_stat.keys()):
        #     col4.metric('Рейтинг поставщика', supp_stat[str(rv)], delta = 'Высокий' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'Низкий', delta_color = 'normal' if supp_stat[str(rv)] in ['5⭐', '4⭐'] else 'inverse')
        # else:
        #     col4.metric('Рейтинг поставщика', 'Неизвестен')



        df = prediction_df.copy()[used_columns]
        shap_df = get_shap_df(df)

        shap_values, explainer = get_shap_values(df)
        feature_names = df.drop(target_col, axis=1).columns
        # feature_names = [cols_le[x] for x in feature_names]

        st.markdown("### 📚 Как читать эти графики?")

        col1, col2 = st.columns(2)

        col1.markdown("""* Значения слева от центральной вертикальной линии — это **negative** класс (0 - Поставка произойдет в срок), справа — **positive** (1 - Поставка будет просрочена) \n* Чем толще линия на графике, тем больше таких точек наблюдения. \n* Чем краснее точки на графике, тем выше значения фичи в ней.""")
        col2.markdown("""* График помогает определить, какие признаки оказывают наибольшее влияние на вероятность задержки поставки и в какую сторону – положительную или отрицательную. \n* Длина столбца - это величина вклада этого фактора. Положительная высота означает, что фактор увеличивает предсказание, а отрицательная - уменьшает. \n* Признаки упорядочены сверху вниз в порядке влияния на предсказание.""")

        # дальше выводим разные графики по полученным shap values (можно указывать height и width аргументами в st_shap)
        col1, col2 = st.columns(2)

        with col1:
            st_shap(shap.summary_plot(shap_values, shap_df.drop(target_col, axis=1), max_display=12, feature_names=feature_names), height=500)

        with col2:
            st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names), height=500)

        st.write('')
        st.write('')
        st.markdown("""###### Влияние признаков на вероятность задержки поставки - длина и количество синих стрелок уменьшает вероятность задержки, длина и количество красных стрелок увеличивает веротность просрочки.""")

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:],
                shap_df.drop(target_col, axis=1).iloc[0,:], feature_names=feature_names))

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots()  # создаем объекты figure и axes
            perm_importance = permutation_importance(model, shap_df.drop(target_col, axis=1), shap_df[target_col], n_repeats=5, random_state=42)
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

            column_to_plot = st.selectbox('Выберите колонку для построения графика:', used_columns)
            st_shap(shap.dependence_plot(column_to_plot, shap_values, shap_df.drop(target_col, axis=1), feature_names=feature_names), height=400)

        st.divider()

        st.markdown('**Корреляционная матрица** - это таблица, отображающая коэффициенты корреляции между различными переменными или признаками в наборе данных. Каждая ячейка в матрице содержит коэффициент корреляции, который измеряет силу и направление связи между двумя соответствующими признаками. Значения ближе к 1 указывают на сильную положительную корреляцию, около 0 указывают на слабую или отсутствующую корреляцию. Аналитики используют корреляционные матрицы для выявления взаимосвязей между переменными и оценки их влияния друг на друга, что полезно для анализа данных и моделирования.')

        phk_df = shap_df.copy()
        phk_df[target_col] = model.predict(shap_df)
        phik_overview = phk_df.phik_matrix(interval_cols=[i for i in phk_df.columns if i != target_col]).round(3).sort_values(target_col)

        data = phik_overview.values
        columns = [cols_le[x] for x in phik_overview.columns]
        index = [cols_le[x] for x in phik_overview.index]

        fig = px.imshow(data, x=columns, y=index)

        fig.update_layout(xaxis_title="", yaxis_title="", height=800)

        fig.update_xaxes(tickfont=dict(size=12), tickangle=45)
        fig.update_yaxes(tickfont=dict(size=12))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('*Для рассчета матрицы используется набор данных похожий на выбранный семпл')

        st.divider()

        # st.button(f'Скачать анализ в PDF', type='primary', use_container_width=True)
        st.download_button('Скачать таблицу с предсказаниями', data=preds, type='secondary', use_container_width=True, file_name='predicted_sales.csv')

        st.divider()



# history
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

    with tab2: 
        st.divider()

        st.dataframe(df, height=200)

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric('Количество уникальных поставщиков', df['Поставщик'].nunique())
        col2.metric('Средняя длительность поставки', round(df['Длительность'].mean(), 2))
        col3.metric('Количество своевременных поставок', df[df[target_col] == 0].shape[0])
        col4.metric('Количество просрочек', df[df[target_col] == 1].shape[0])

        st.divider()
        
        st.markdown('##### Анализ предсказания модели на тренировочной выборке')

        col1, col2 = st.columns(2)
        
        preds_train = pd.read_csv('preds_train.csv')
        
        y_true = preds_train["gt"]
        y_pred = preds_train["pred"]
        

        class_report = classification_report(y_true, y_pred, target_names=["В срок", "Просрочка"], output_dict=True)

        conf_matrix = confusion_matrix(y_true, y_pred)

        with col1:
            # Convert the classification report to a DataFrame
            df_report = pd.DataFrame(class_report).apply(lambda x: round(x, 3)).transpose()

            # Create a Plotly table
            fig = go.Figure(data=[go.Table(
                header=dict(values=['', 'Precision', 'Recall', 'F1-Score', 'Support'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[df_report.index, 
                                df_report['precision'], 
                                df_report['recall'], 
                                df_report['f1-score'], 
                                df_report['support']],
                        # fill_color='lavender',
                        align='center'))
            ])

            fig.update_layout(title='Классификационные метрики модели для тренировочной выборки', margin=dict(b=0), autosize=False, height=250)

            # Show the plot
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                Каждая метрика отчете имеет свою специфическую интерпретацию:

                * **Точность (Precision):** Показывает, какая доля положительных предсказаний была правильно классифицирована; то есть, сколько из обнаруженных как положительные действительно принадлежат к целевому классу.

                * **Полнота (Recall):** Отражает, какая доля истинно положительных случаев была успешно найдена моделью; она измеряет, сколько из всех фактически положительных примеров было обнаружено.

                * **F1-мера (F1-Score):** Это гармоническое среднее между точностью и полнотой. Она помогает балансировать между этими двумя метриками, особенно если они имеют разные веса в задаче.

                * **Поддержка (Support):** Это количество примеров в каждом классе. Это полезно для понимания, сколько примеров на самом деле составляют каждый класс.
                """
                )

        with col2:
            fig = px.imshow(conf_matrix,
                labels=dict(x="Предсказание", y="Истина"),
                x=["В срок", "Просрочка"],
                y=["В срок", "Просрочка"],
                title="Матрица ошибок")
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix[0])):
                    fig.add_annotation(
                        text=str(conf_matrix[i][j]),
                        x=["В срок", "Просрочка"][j],
                        y=["В срок", "Просрочка"][i],
                        showarrow=False,
                        font=dict(color="black"),
                        xref="x",
                        yref="y",
                        xanchor="center",
                        yanchor="middle",
                    )
            fig.update_xaxes(side="top")

            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('* Интерпретация матрицы ошибок помогает понять, \
                         насколько хорошо модель справляется с определенной задачей и может помочь в оптимизации модели или настройке пороговых значений для более точных предсказаний.')
        
        st.divider()
        
        st.markdown('##### Анализ данных')

        col1, col2 = st.columns(2)

        df['supp_rating'] = df['Поставщик'].apply(lambda x: supp_stat[str(x)])

        with col1:
            rating_y_count = df.groupby(['supp_rating'])[target_col].value_counts().reset_index(name='count')

            fig = px.bar(rating_y_count, x='supp_rating', y='count', color=rating_y_count[target_col].astype(str), 
                        labels={'count': 'Количество', 'supp_rating': 'Рейтинг', 'color': 'Статус поставки'}, barmode='group',
                        title='Количество своевременных/просроченных поставок для каждого рейтинга поставщика.')

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            rating_len_mean = df.groupby(['supp_rating'])['Длительность'].mean().reset_index(name='mean')


            fig = px.bar(rating_len_mean, x='supp_rating', y='mean', color='supp_rating', 
                        labels={'mean': 'Средняя длительность', 'supp_rating': 'Рейтинг', 'color': 'Рейтинг'},
                        title='Средняя длительность поставки для каждого рейтинга поставщика')

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        supplier_to_plot = st.selectbox('Выберите поставщика для построения графиков:', sorted(df['Поставщик'].unique()))

        col1, col2 = st.columns(2)

        with col1:
            supp_y_count = df.groupby(['Поставщик'])[target_col].value_counts()

            fig = px.bar(x=[supplier_to_plot, supplier_to_plot], y=supp_y_count[supplier_to_plot].values,
                         color=supp_y_count[supplier_to_plot].index.astype(str), barmode='group', 
                         labels={'x': 'Поставщик', 'y': 'Количество', 'color': 'Статус поставки'}, 
                         title=f'Количество своевременных/просроченных поставок для поставщика {supplier_to_plot}')
            st.plotly_chart(fig, use_container_width=True)



        with col2:
            supp_mat_count = df.groupby(['Поставщик'])['Материал'].value_counts()

            fig = px.pie(values=supp_mat_count[supplier_to_plot][:10].values, names=supp_mat_count[supplier_to_plot][:10].index, title=f'Топ 10 материалов по количеству для поставщика {supplier_to_plot}')
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)


        col1, col2 = st.columns(2)


        with col1:
            fig = ff.create_distplot([df[(df['Поставщик'] == supplier_to_plot) & (df[target_col] == 0)]['Длительность'].values, 
                                      df[(df['Поставщик'] == supplier_to_plot) & (df[target_col] == 1)]['Длительность'].values], 
                                     [f'Поставщик = {supplier_to_plot} | Статус поставки = В срок', f'Поставщик = {supplier_to_plot} | Статус поставки = Просрочка'])
            fig.update_layout(title_text=f'График распределения длительности поставки для поставщика {supplier_to_plot}, группируя по статусу поставки')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()


            # выбор колонки
            column_for_time = st.selectbox('Выберите колонку', cat_fts)
            unique_values = sorted(df[column_for_time].unique().tolist())
            # выбор конкретного значения из уникальных значений колонки
            selected_value = st.selectbox('Выберите значение', unique_values)
            st.plotly_chart(plot_failures_over_time(df, column_for_time, selected_value))

        with col2:
            sup_mat_y = df.groupby(['Поставщик', 'Материал'])[target_col].value_counts()

            x = sup_mat_y[supplier_to_plot][supp_mat_count[supplier_to_plot][:10].index].reset_index(name='y_count')
            x[['Материал', target_col]] = x[['Материал', target_col]].astype(str)
            x['Материал'] = x['Материал'] + '_'

            fig = px.bar(x, x='Материал', y='y_count',
                        color=target_col, barmode='group', 
                        labels={'y_count': 'Количество', target_col: 'Статус поставки'}, 
                        title=f'Количество своевременных/просроченных поставок для топа материалов поставщика {supplier_to_plot}')
            st.plotly_chart(fig, use_container_width=True)


            st.divider()

            columns_to_group = st.multiselect('Выберите колонки для группировки', cat_fts, default=['Поставщик'])
            selected_values = {}

            # Создаем виджеты выбора значений для каждой колонки
            for column in columns_to_group:
                unique_values = sorted(df[column].unique().tolist())
                selected_values[column] = st.selectbox(f"Выберите значение для '{column}'", unique_values)

            if selected_values:
                filtered_samples = get_filtered_samples(df, selected_values)
                rating_y_count = filtered_samples[target_col].value_counts().reset_index(name='count').rename({'index':target_col}, axis=1)

                fig = px.bar(rating_y_count, x=target_col, y='count', 
                    labels={'count': 'Количество'}, barmode='group',
                    title='Количество своевременных/просроченных поставок для колонок с выбранными значениями')

                st.plotly_chart(fig, use_container_width=True)

        if selected_values:

            col1, col2 = st.columns(2)

            with col1:
                values_to_plot = filtered_samples.groupby(target_col)[['Количество обработчиков 7', 'Количество обработчиков 15', 'Количество обработчиков 30']].mean().reset_index().melt(target_col)
                values_to_plot['variable'] = [7, 7, 15, 15, 30, 30]
                fig = px.line(values_to_plot, x='variable', y='value', color=target_col, title=f'Динамика изменения количества обработчиков')
                fig.update_traces(mode='markers+lines')
                fig.update_xaxes(title='Дни')
                fig.update_yaxes(title='Среднее к-во изменений обработчиков')
                st.plotly_chart(fig, use_container_width=True)

            with col2:

                values_to_plot = filtered_samples.groupby(target_col)[['Изменение даты поставки 7','Изменение даты поставки 15','Изменение даты поставки 30']].mean().reset_index().melt(target_col)
                values_to_plot['variable'] = [7, 7, 15, 15, 30, 30]
                fig = px.line(values_to_plot, x='variable', y='value', color=target_col, title=f'Динамика изменения дат поставки')
                fig.update_traces(mode='markers+lines')
                fig.update_xaxes(title='Дни')
                fig.update_yaxes(title='Среднее к-во изменений дат поставки')
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.button('Скачать Исторический Анализ в PDF', type='primary', use_container_width=True)

        st.divider()
