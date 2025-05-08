# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Можно заменить на LogisticRegression для скорости
from sklearn.metrics import accuracy_score, classification_report
from pmdarima import auto_arima
import matplotlib.pyplot as plt # pmdarima может использовать matplotlib для summary

# --- 0. Настройки страницы и утилиты ---
st.set_page_config(layout="wide", page_title="Анализ ДТП")

# Кэширование загрузки данных
@st.cache_data
def load_data(file_path="Road Accident Data.csv"):
    df = pd.read_csv(file_path)
    df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
    # Удаляем Accident_Index, если есть, игнорируя ошибку, если уже удален
    if 'Accident_Index' in df.columns:
        df = df.drop(['Accident_Index'], axis=1)
    return df

# Кэширование подготовки данных для ML (целевая переменная и признаки)
@st.cache_data
def prepare_ml_data(_df, sample_frac=0.1): # Используем _df чтобы streamlit отслеживал изменения
    df_ml = _df.copy()

    # Упрощенная обработка времени для Streamlit - извлечение часа
    if 'Time' in df_ml.columns:
        df_ml['Time'] = df_ml['Time'].astype(str)
        df_ml['Hour'] = pd.to_datetime(df_ml['Time'], format='%H:%M', errors='coerce').dt.hour
        df_ml['Hour'].fillna(df_ml['Hour'].mode()[0], inplace=True)
        df_ml.drop('Time', axis=1, inplace=True)

    if 'Accident Date' in df_ml.columns:
        df_ml.drop('Accident Date', axis=1, inplace=True) # Уже не нужна для этой модели

    # Убираем Number_of_Casualties - потенциальный лик данных
    if 'Number_of_Casualties' in df_ml.columns:
        df_ml.drop(['Number_of_Casualties'], axis=1, inplace=True)

    # Для скорости в Streamlit возьмем выборку
    if sample_frac < 1.0:
        df_ml = df_ml.sample(frac=sample_frac, random_state=42)

    # Убираем строки с NaN в целевой переменной
    df_ml.dropna(subset=['Accident_Severity'], inplace=True)
    
    y_raw = df_ml['Accident_Severity']
    X = df_ml.drop('Accident_Severity', axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    return X, y, label_encoder, df_ml.columns # Возвращаем колонки для использования в UI

# Кэширование обучения модели классификации
# @st.cache_data # Не кэшируем, т.к. _preprocessor может содержать OneHotEncoder, который плохо кэшируется напрямую
# Вместо этого можно кэшировать сам объект pipeline с моделью через @st.cache_resource
@st.cache_resource # Для объектов, которые нельзя хешировать стандартными методами data
def get_trained_severity_model(X_train, y_train, numerical_features, categorical_features):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')

    # Используем LogisticRegression для скорости в Streamlit
    # model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1, max_depth=10)
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=300)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

# Кэширование подготовки данных для временных рядов
@st.cache_data
def get_time_series_data(_df):
    df_ts = _df[['Accident Date']].copy()
    df_ts.dropna(subset=['Accident Date'], inplace=True)
    df_ts.set_index('Accident Date', inplace=True)
    monthly_accidents = df_ts.resample('M').size().rename('Num_Accidents')
    return monthly_accidents

# Кэширование обучения модели временных рядов
@st.cache_resource
def get_trained_sarima_model(train_data):
    if len(train_data) < 24: # Минимальное условие для m=12
        st.warning("Данных очень мало для надежной модели SARIMA с годовой сезонностью. Могут использоваться упрощенные параметры.")
        # Упрощенные параметры, если данных мало
        try:
            sarima_model = auto_arima(train_data,
                                  start_p=1, start_q=1, max_p=2, max_q=2,
                                  start_P=0, start_Q=0, max_P=1, max_Q=1,
                                  m=max(1, min(12, len(train_data)//2)), # Адаптивная сезонность, если мало данных
                                  seasonal=True if len(train_data) > 12 else False, # Сезонность если > года данных
                                  stepwise=True, suppress_warnings=True,
                                  error_action='ignore', trace=False)
        except Exception as e: # Если auto_arima падает на малых данных
             st.error(f"Ошибка при обучении auto_arima: {e}. Используется очень простая модель.")
             # Очень простая ARIMA(1,0,0) как fallback
             from statsmodels.tsa.arima.model import ARIMA
             sarima_model = ARIMA(train_data, order=(1,0,0)).fit()
    else:
        sarima_model = auto_arima(train_data,
                                  start_p=1, start_q=1, max_p=3, max_q=3,
                                  start_P=0, start_Q=0, max_P=2, max_Q=2,
                                  m=12, seasonal=True, stepwise=True,
                                  suppress_warnings=True, D=1, error_action='ignore', trace=False)
    return sarima_model

# --- 1. Загрузка данных ---
df_main = load_data()

# --- Боковая панель для навигации ---
st.sidebar.title("Навигация")
app_mode = st.sidebar.selectbox("Выберите раздел:",
                                ["Обзор данных", "Исследовательский анализ (EDA)",
                                 "Прогнозирование серьезности ДТП", "Прогнозирование количества ДТП (Временные ряды)"])

# --- 2. Раздел "Обзор данных" ---
if app_mode == "Обзор данных":
    st.title("Обзор данных о ДТП")
    st.write("Первые несколько строк данных:")
    st.dataframe(df_main.head())

    st.write("Размер датасета:")
    st.write(df_main.shape)

    st.write("Типы данных:")
    st.dataframe(df_main.dtypes.astype(str).to_frame(name='Тип данных'))

    st.write("Описательная статистика (числовые признаки):")
    st.dataframe(df_main.describe(include=np.number))
    
    st.write("Описательная статистика (категориальные признаки):")
    st.dataframe(df_main.describe(include='object'))

    st.write("Пропущенные значения:")
    missing_values = df_main.isnull().sum()
    missing_df = missing_values[missing_values > 0].sort_values(ascending=False).to_frame(name='Кол-во пропусков')
    st.dataframe(missing_df)

# --- 3. Раздел "Исследовательский анализ (EDA)" ---
elif app_mode == "Исследовательский анализ (EDA)":
    st.title("Исследовательский анализ данных (EDA)")

    # Распределение серьезности ДТП
    st.subheader("Распределение серьезности ДТП")
    severity_counts = df_main['Accident_Severity'].value_counts()
    fig_severity = px.bar(severity_counts, x=severity_counts.index, y=severity_counts.values,
                          labels={'x': 'Серьезность ДТП', 'y': 'Количество'},
                          title="Распределение серьезности ДТП")
    st.plotly_chart(fig_severity, use_container_width=True)

    # ДТП по дням недели
    st.subheader("Количество ДТП по дням недели")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if 'Day_of_Week' in df_main.columns:
        day_counts = df_main['Day_of_Week'].value_counts().reindex(day_order)
        fig_day = px.bar(day_counts, x=day_counts.index, y=day_counts.values,
                         labels={'x': 'День недели', 'y': 'Количество ДТП'},
                         title="ДТП по дням недели")
        st.plotly_chart(fig_day, use_container_width=True)

    # ДТП по месяцам
    st.subheader("Количество ДТП по месяцам")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if 'Month' in df_main.columns:
        month_counts = df_main['Month'].value_counts().reindex(month_order, fill_value=0) # fill_value=0 если какого-то месяца нет
        fig_month = px.bar(month_counts, x=month_counts.index, y=month_counts.values,
                           labels={'x': 'Месяц', 'y': 'Количество ДТП'},
                           title="ДТП по месяцам")
        st.plotly_chart(fig_month, use_container_width=True)

    # ДТП по погодным условиям (Топ N)
    st.subheader("Топ N погодных условий при ДТП")
    if 'Weather_Conditions' in df_main.columns:
        top_n_weather = st.slider("Выберите N для погодных условий:", 3, 10, 5)
        weather_counts = df_main['Weather_Conditions'].value_counts().nlargest(top_n_weather)
        fig_weather = px.bar(weather_counts, x=weather_counts.index, y=weather_counts.values,
                             labels={'x': 'Погодные условия', 'y': 'Количество ДТП'},
                             title=f"Топ-{top_n_weather} погодных условий")
        st.plotly_chart(fig_weather, use_container_width=True)

    # Карта ДТП (если есть координаты и они числовые)
    st.subheader("Карта ДТП (выборка для производительности)")
    if 'Latitude' in df_main.columns and 'Longitude' in df_main.columns:
        map_df = df_main[['Latitude', 'Longitude']].copy()
        map_df.dropna(inplace=True) # Удаляем строки без координат
        # Убедимся, что типы числовые
        map_df['Latitude'] = pd.to_numeric(map_df['Latitude'], errors='coerce')
        map_df['Longitude'] = pd.to_numeric(map_df['Longitude'], errors='coerce')
        map_df.dropna(inplace=True)

        if not map_df.empty:
            st.map(map_df.sample(min(10000, len(map_df)))) # Отображаем выборку для производительности
        else:
            st.write("Недостаточно данных с координатами для отображения карты.")
    else:
        st.write("Колонки 'Latitude' и 'Longitude' отсутствуют или содержат некорректные данные.")


# --- 4. Раздел "Прогнозирование серьезности ДТП" ---
elif app_mode == "Прогнозирование серьезности ДТП":
    st.title("Прогнозирование серьезности ДТП")
    
    with st.spinner("Подготовка данных и обучение модели... Это может занять некоторое время."):
        # Используем 30% данных для обучения в Streamlit, чтобы было быстрее
        X, y, label_encoder, original_cols = prepare_ml_data(df_main.copy(), sample_frac=0.3)

        # Определение типов признаков на основе подготовленного X
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Убедимся, что Hour попал куда нужно
        if 'Hour' in X.columns and 'Hour' not in numerical_features and 'Hour' not in categorical_features:
            if pd.api.types.is_numeric_dtype(X['Hour']):
                if 'Hour' not in numerical_features: numerical_features.append('Hour')
            else:
                if 'Hour' not in categorical_features: categorical_features.append('Hour')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Проверка, что списки признаков не пустые
        valid_num_features = [f for f in numerical_features if f in X_train.columns]
        valid_cat_features = [f for f in categorical_features if f in X_train.columns]

        if not valid_num_features and not valid_cat_features:
            st.error("Не удалось определить числовые или категориальные признаки для модели. Проверьте данные.")
        else:
            ml_pipeline = get_trained_severity_model(X_train, y_train, valid_num_features, valid_cat_features)

    st.subheader("Интерактивное предсказание")
    st.write("Введите данные для предсказания серьезности ДТП (некоторые поля могут быть недоступны, если их нет в обучающей выборке после очистки).")

    input_data = {}
    cols = st.columns(3)
    col_idx = 0

    # Собираем признаки, которые были в X перед train_test_split
    # Это важно, так как X_train может иметь меньше колонок из-за выборки
    available_features_for_input = X.columns 
    
    for feature in available_features_for_input:
        with cols[col_idx % 3]:
            if feature in valid_num_features:
                # Для числовых признаков (кроме 'Hour' и 'Year' если они есть)
                if feature == 'Hour':
                     input_data[feature] = st.slider(f"{feature}", 0, 23, X[feature].median() if not X[feature].empty else 12)
                elif feature == 'Year': # Если 'Year' остался как числовой
                    years = sorted(X[feature].unique().astype(int))
                    input_data[feature] = st.selectbox(f"{feature}", years, index = len(years)//2 if years else 0)
                elif not X[feature].empty:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    median_val = float(X[feature].median())
                    input_data[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=median_val, step=(max_val-min_val)/100 if max_val != min_val else 0.1)
                else: # Если колонка пуста после prepare_ml_data
                    input_data[feature] = st.number_input(f"{feature} (нет данных в выборке)", value=0.0)

            elif feature in valid_cat_features:
                unique_vals = list(X[feature].unique())
                if X[feature].empty or not unique_vals : # Если колонка пуста или нет уникальных значений
                     input_data[feature] = st.text_input(f"{feature} (нет данных в выборке)", value="N/A")
                else:
                    default_val = X[feature].mode()[0] if not X[feature].empty else unique_vals[0]
                    try: # Попытка найти индекс моды, если нет - берем первый
                        default_ix = unique_vals.index(default_val)
                    except ValueError:
                        default_ix = 0
                    input_data[feature] = st.selectbox(f"{feature}", unique_vals, index=default_ix)
            col_idx += 1
    
    if st.button("Предсказать серьезность"):
        if 'ml_pipeline' in locals():
            try:
                # Создаем DataFrame из введенных данных
                # Убедимся, что порядок колонок соответствует X_train
                input_df = pd.DataFrame([input_data])[list(X_train.columns)]

                prediction_encoded = ml_pipeline.predict(input_df)
                prediction_proba = ml_pipeline.predict_proba(input_df)
                
                predicted_severity = label_encoder.inverse_transform(prediction_encoded)[0]
                st.success(f"Предсказанная серьезность ДТП: **{predicted_severity}**")

                st.subheader("Вероятности классов:")
                proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
                st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
                st.error("Возможно, некоторые из введенных значений отсутствовали в обучающих данных или проблема с форматом.")
        else:
            st.error("Модель еще не обучена. Пожалуйста, подождите или проверьте консоль на ошибки.")

    st.subheader("Оценка модели (на тестовой выборке из 30% данных)")
    if 'ml_pipeline' in locals() and not X_test.empty:
        y_pred_test = ml_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        st.write(f"Точность модели: {accuracy:.4f}")
        
        report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())
    elif X_test.empty:
         st.warning("Тестовая выборка пуста, оценка не производится.")


# --- 5. Раздел "Прогнозирование количества ДТП (Временные ряды)" ---
elif app_mode == "Прогнозирование количества ДТП (Временные ряды)":
    st.title("Прогнозирование количества ДТП (Временные ряды)")

    monthly_accidents = get_time_series_data(df_main.copy())

    st.subheader("Ежемесячное количество ДТП")
    fig_ts = px.line(monthly_accidents, x=monthly_accidents.index, y='Num_Accidents',
                     labels={'Num_Accidents': 'Количество ДТП', 'Accident Date': 'Дата'},
                     title="Ежемесячное количество ДТП")
    fig_ts.update_layout(xaxis_title='Дата', yaxis_title='Количество ДТП')
    st.plotly_chart(fig_ts, use_container_width=True)

    if len(monthly_accidents) > 3: # Минимальное количество данных для разделения
        train_data_ts = monthly_accidents[:-3] # Оставляем 3 месяца для теста
        test_data_ts = monthly_accidents[-3:]
    else:
        train_data_ts = monthly_accidents
        test_data_ts = pd.Series(dtype='float64') # Пустая серия, если мало данных

    if not train_data_ts.empty:
        with st.spinner("Обучение модели SARIMA... Это может занять некоторое время."):
            sarima_model = get_trained_sarima_model(train_data_ts)

        st.subheader("Сводка по модели SARIMA")
        # Pmdarima summary может выводить в консоль, перехватим это
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            sarima_model.summary() # Это может быть statsmodels summary, если pmdarima его вернул
            summary_text = captured_output.getvalue()
        except AttributeError: # Если это ARIMA из statsmodels напрямую (fallback)
            summary_text = sarima_model.summary().as_text()
        finally:
            sys.stdout = old_stdout # Возвращаем stdout
        st.text(summary_text)

        st.subheader("Прогноз количества ДТП")
        n_periods_forecast = st.slider("Количество месяцев для прогноза:", 1, 24, 6)

        # Прогноз включает и тестовый период (если есть) и будущие периоды
        if not test_data_ts.empty:
            total_forecast_periods = len(test_data_ts) + n_periods_forecast
            forecast_values_with_conf = sarima_model.predict(n_periods=total_forecast_periods, return_conf_int=True)
            forecast_values = forecast_values_with_conf[0]
            conf_int = forecast_values_with_conf[1]
            
            # Индекс для всего прогноза
            forecast_index_start = test_data_ts.index[0]
        else: # Если тестовых данных нет, прогнозируем с конца обучающих
            total_forecast_periods = n_periods_forecast
            forecast_values_with_conf = sarima_model.predict(n_periods=total_forecast_periods, return_conf_int=True)
            forecast_values = forecast_values_with_conf[0]
            conf_int = forecast_values_with_conf[1]
            if not train_data_ts.empty:
                forecast_index_start = train_data_ts.index[-1] + pd.DateOffset(months=1)
            else: # Крайний случай, если train_data_ts тоже пуст
                st.warning("Нет данных для начала прогноза.")
                forecast_index_start = pd.to_datetime('today') + pd.DateOffset(months=1)


        forecast_idx = pd.date_range(start=forecast_index_start, periods=total_forecast_periods, freq='M')
        
        forecast_series = pd.Series(forecast_values, index=forecast_idx)
        lower_bound = pd.Series(conf_int[:, 0], index=forecast_idx)
        upper_bound = pd.Series(conf_int[:, 1], index=forecast_idx)

        fig_forecast = plt.figure(figsize=(12, 6))
        ax = fig_forecast.gca()
        monthly_accidents.plot(ax=ax, label='Исторические данные')
        if not test_data_ts.empty:
             test_data_ts.plot(ax=ax, label='Реальные значения (тест)', style='.')
        forecast_series.plot(ax=ax, label='Прогноз SARIMA')
        ax.fill_between(forecast_idx, lower_bound, upper_bound, color='pink', alpha=0.3, label='Доверительный интервал 95%')
        plt.title('Прогноз количества ДТП с помощью SARIMA')
        plt.xlabel('Дата')
        plt.ylabel('Количество ДТП')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_forecast)
        
        if not test_data_ts.empty:
            st.write("Прогноз на тестовом периоде:")
            test_forecast = forecast_series.iloc[:len(test_data_ts)]
            comparison_df = pd.DataFrame({'Реальные': test_data_ts, 'Прогноз': test_forecast.round(2)})
            st.dataframe(comparison_df)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(test_data_ts, test_forecast)
            rmse = np.sqrt(mean_squared_error(test_data_ts, test_forecast))
            st.write(f"MAE на тестовых данных: {mae:.2f}")
            st.write(f"RMSE на тестовых данных: {rmse:.2f}")

    else:
        st.warning("Недостаточно данных для построения модели временных рядов.")

st.sidebar.markdown("---")
st.sidebar.info("Разработано на основе анализа данных ДТП.")
