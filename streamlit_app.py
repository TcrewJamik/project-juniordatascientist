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
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error # Глобальный импорт метрик
from pmdarima import auto_arima
import matplotlib.pyplot as plt # pmdarima может использовать matplotlib для summary
import zipfile # Импорт для работы с ZIP
import io      # Импорт для работы с байтами в памяти (может не понадобиться напрямую с pd.read_csv)

# --- 0. Настройки страницы и утилиты ---
st.set_page_config(layout="wide", page_title="Анализ ДТП")

# Кэширование загрузки данных (версия для чтения из ZIP)
@st.cache_data
def load_data(zip_file_path="Road Accident Data.zip", csv_file_in_zip="Road Accident Data.csv"):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            csv_filename = None
            # Ищем CSV файл внутри архива, чтобы быть гибче к его точному имени/пути внутри ZIP
            for name_in_zip in z.namelist():
                if name_in_zip.lower().endswith('.csv') and "macOS" not in name_in_zip.lower() and not name_in_zip.startswith('__MACOSX'):
                    csv_filename = name_in_zip
                    break 

            if csv_filename is None:
                if csv_file_in_zip in z.namelist():
                    csv_filename = csv_file_in_zip
                else:
                    st.error(f"CSV файл ('{csv_file_in_zip}' или любой другой .csv) не найден внутри архива {zip_file_path}")
                    st.info(f"Файлы в архиве: {z.namelist()}")
                    return pd.DataFrame()

            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
                
        df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
        if 'Accident_Index' in df.columns:
            df = df.drop(['Accident_Index'], axis=1)
        st.success(f"Данные успешно загружены из '{csv_filename}' внутри архива '{zip_file_path}'")
        return df
    except FileNotFoundError:
        st.error(f"Файл архива '{zip_file_path}' не найден. Убедитесь, что он находится в той же директории, что и app.py, или в корне репозитория.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ошибка при загрузке или обработке данных из ZIP: {e}")
        return pd.DataFrame()

# Кэширование подготовки данных для ML (целевая переменная и признаки)
@st.cache_data
def prepare_ml_data(_df, sample_frac=0.1):
    if _df.empty:
        st.warning("Входной DataFrame для prepare_ml_data пуст. Пропускаем подготовку данных для ML.")
        return pd.DataFrame(), pd.Series(dtype='int'), None, [] # Возвращаем None для label_encoder

    df_ml = _df.copy()

    if 'Time' in df_ml.columns:
        df_ml['Time'] = df_ml['Time'].astype(str)
        df_ml['Hour'] = pd.to_datetime(df_ml['Time'], format='%H:%M', errors='coerce').dt.hour
        df_ml['Hour'].fillna(df_ml['Hour'].mode()[0], inplace=True) # Используем mode()[0] для заполнения
        df_ml.drop('Time', axis=1, inplace=True)

    if 'Accident Date' in df_ml.columns:
        df_ml.drop('Accident Date', axis=1, inplace=True) 

    if 'Number_of_Casualties' in df_ml.columns:
        df_ml.drop(['Number_of_Casualties'], axis=1, inplace=True)

    if sample_frac < 1.0 and len(df_ml) > 0 :
        df_ml = df_ml.sample(frac=sample_frac, random_state=42)

    if 'Accident_Severity' not in df_ml.columns:
        st.error("Колонка 'Accident_Severity' отсутствует в данных после начальной обработки для ML.")
        return pd.DataFrame(), pd.Series(dtype='int'), None, []

    df_ml.dropna(subset=['Accident_Severity'], inplace=True)
    if df_ml.empty:
        st.warning("DataFrame стал пустым после удаления NaN в 'Accident_Severity'.")
        return pd.DataFrame(), pd.Series(dtype='int'), None, []
    
    y_raw = df_ml['Accident_Severity']
    X = df_ml.drop('Accident_Severity', axis=1)

    if y_raw.empty:
        st.warning("Целевая переменная y_raw пуста перед кодированием.")
        return X, pd.Series(dtype='int'), None, X.columns.tolist()


    label_encoder = LabelEncoder()
    try:
        y = label_encoder.fit_transform(y_raw)
        y = pd.Series(y, name='Accident_Severity_Encoded', index=X.index if X.index.equals(y_raw.index) else None) # Сохраняем индекс если возможно
    except Exception as e:
        st.error(f"Ошибка при кодировании целевой переменной: {e}")
        return X, pd.Series(dtype='int'), None, X.columns.tolist()
    
    return X, y, label_encoder, df_ml.columns.tolist() # Возвращаем df_ml.columns

# Кэширование обучения модели классификации
@st.cache_resource
def get_trained_severity_model(X_train, y_train, numerical_features, categorical_features):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_pipeline, categorical_features))
    
    if not transformers:
        st.error("Нет ни числовых, ни категориальных признаков для preprocessor.")
        return None 
        
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=300)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Ошибка при обучении модели классификации: {e}")
        return None
    return pipeline

# Кэширование подготовки данных для временных рядов
@st.cache_data
def get_time_series_data(_df):
    if _df.empty or 'Accident Date' not in _df.columns:
        st.warning("Входной DataFrame для get_time_series_data пуст или не содержит 'Accident Date'.")
        return pd.Series(dtype='float64')

    df_ts = _df[['Accident Date']].copy()
    df_ts.dropna(subset=['Accident Date'], inplace=True)
    if df_ts.empty:
        st.warning("DataFrame для временных рядов стал пустым после удаления NaN в 'Accident Date'.")
        return pd.Series(dtype='float64')

    df_ts.set_index('Accident Date', inplace=True)
    monthly_accidents = df_ts.resample('M').size().rename('Num_Accidents')
    return monthly_accidents

# Кэширование обучения модели временных рядов
@st.cache_resource
def get_trained_sarima_model(train_data):
    if train_data.empty:
        st.warning("Нет данных для обучения модели SARIMA.")
        return None

    m_param = 12
    seasonal_param = True
    D_param = 1 

    if len(train_data) < 2 * m_param : 
        st.warning(f"Данных ({len(train_data)} точек) очень мало для годовой сезонности (m=12). Сезонность может быть отключена или m уменьшено.")
        if len(train_data) < 12 :
            seasonal_param = False
            m_param = 1 
            D_param = None 
        else: 
            m_param = max(1, len(train_data) // 2) 
    
    try:
        sarima_model = auto_arima(train_data,
                              start_p=1, start_q=1, max_p=2, max_q=2,
                              start_P=0, start_Q=0, max_P=1, max_Q=1, 
                              m=m_param,
                              seasonal=seasonal_param,
                              D=D_param if seasonal_param else None, 
                              stepwise=True, suppress_warnings=True,
                              error_action='ignore', trace=False,
                              maxiter=50) 
    except Exception as e:
         st.error(f"Ошибка при обучении auto_arima: {e}. Попытка с более простой моделью.")
         from statsmodels.tsa.arima.model import ARIMA
         try:
            sarima_model = ARIMA(train_data, order=(1,1,0)).fit()
         except Exception as e_arima:
            st.error(f"Ошибка при обучении fallback ARIMA: {e_arima}")
            return None
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
    if not df_main.empty:
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
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.write("Пропущенных значений не найдено.")
    else:
        st.warning("Данные не загружены. Проверьте сообщения об ошибках выше.")


# --- 3. Раздел "Исследовательский анализ (EDA)" ---
elif app_mode == "Исследовательский анализ (EDA)":
    st.title("Исследовательский анализ данных (EDA)")
    if not df_main.empty:
        if 'Accident_Severity' in df_main.columns:
            st.subheader("Распределение серьезности ДТП")
            severity_counts = df_main['Accident_Severity'].value_counts()
            fig_severity = px.bar(severity_counts, x=severity_counts.index, y=severity_counts.values,
                                  labels={'x': 'Серьезность ДТП', 'y': 'Количество'},
                                  title="Распределение серьезности ДТП")
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.warning("Колонка 'Accident_Severity' отсутствует в данных.")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if 'Day_of_Week' in df_main.columns:
            st.subheader("Количество ДТП по дням недели")
            day_counts = df_main['Day_of_Week'].value_counts().reindex(day_order).fillna(0)
            fig_day = px.bar(day_counts, x=day_counts.index, y=day_counts.values,
                             labels={'x': 'День недели', 'y': 'Количество ДТП'},
                             title="ДТП по дням недели")
            st.plotly_chart(fig_day, use_container_width=True)

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if 'Month' in df_main.columns:
            st.subheader("Количество ДТП по месяцам")
            month_counts = df_main['Month'].value_counts().reindex(month_order, fill_value=0)
            fig_month = px.bar(month_counts, x=month_counts.index, y=month_counts.values,
                               labels={'x': 'Месяц', 'y': 'Количество ДТП'},
                               title="ДТП по месяцам")
            st.plotly_chart(fig_month, use_container_width=True)

        if 'Weather_Conditions' in df_main.columns:
            st.subheader("Топ N погодных условий при ДТП")
            unique_weather_count = df_main['Weather_Conditions'].nunique()
            top_n_weather = st.slider("Выберите N для погодных условий:", 1, min(10, unique_weather_count), min(5, unique_weather_count)) # Изменил min и default
            weather_counts = df_main['Weather_Conditions'].value_counts().nlargest(top_n_weather)
            fig_weather = px.bar(weather_counts, x=weather_counts.index, y=weather_counts.values,
                                 labels={'x': 'Погодные условия', 'y': 'Количество ДТП'},
                                 title=f"Топ-{top_n_weather} погодных условий")
            st.plotly_chart(fig_weather, use_container_width=True)

        st.subheader("Карта ДТП (выборка для производительности)")
        if 'Latitude' in df_main.columns and 'Longitude' in df_main.columns:
            map_df = df_main.copy()
            map_df['lat'] = pd.to_numeric(map_df['Latitude'], errors='coerce')
            map_df['lon'] = pd.to_numeric(map_df['Longitude'], errors='coerce')
            map_df.dropna(subset=['lat', 'lon'], inplace=True)

            if not map_df.empty:
                st.map(map_df[['lat', 'lon']].sample(min(10000, len(map_df))))
            else:
                st.write("Недостаточно данных с корректными координатами для отображения карты.")
        else:
            st.write("Колонки 'Latitude' и/или 'Longitude' отсутствуют в исходных данных.")
    else:
        st.warning("Данные не загружены. EDA не может быть отображен.")


# --- 4. Раздел "Прогнозирование серьезности ДТП" ---
elif app_mode == "Прогнозирование серьезности ДТП":
    st.title("Прогнозирование серьезности ДТП")
    
    ml_data_prepared_successfully = False 
    if df_main.empty:
        st.error("Данные не загружены. Невозможно продолжить с прогнозированием серьезности.")
    else:
        with st.spinner("Подготовка данных и обучение модели... Это может занять некоторое время."):
            X, y, label_encoder, original_cols_ml = prepare_ml_data(df_main.copy(), sample_frac=0.3)

        if isinstance(X, pd.DataFrame) and not X.empty and \
           isinstance(y, pd.Series) and not y.empty and label_encoder is not None:
            if y.notna().any(): 
                ml_data_prepared_successfully = True
            else:
                st.error("Данные для целевой переменной (y) пусты или содержат только NaN после кодирования.")
        else:
            st.error("Не удалось корректно подготовить данные для ML (X, y или label_encoder некорректны). Проверьте входные данные и функцию prepare_ml_data.")
    
    if ml_data_prepared_successfully:
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'Hour' in X.columns and 'Hour' not in numerical_features and 'Hour' not in categorical_features:
            if pd.api.types.is_numeric_dtype(X['Hour']):
                if 'Hour' not in numerical_features: numerical_features.append('Hour')
            elif 'Hour' not in categorical_features: categorical_features.append('Hour')
        
        # X_train, X_test, y_train, y_test инициализируются здесь, чтобы были доступны ниже
        X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='int')
        ml_pipeline = None # Инициализируем ml_pipeline

        if not X.empty and not y.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            
            valid_num_features = [f for f in numerical_features if f in X_train.columns]
            valid_cat_features = [f for f in categorical_features if f in X_train.columns]

            if not valid_num_features and not valid_cat_features and not X_train.empty : # Добавил X_train.empty
                st.error("Не удалось определить числовые или категориальные признаки для модели после разделения. Проверьте данные.")
            elif X_train.empty : # Если X_train пуст после разделения (маловероятно, но возможно при малом X)
                 st.error("Обучающая выборка X_train пуста после разделения.")
            else:
                ml_pipeline = get_trained_severity_model(X_train, y_train, valid_num_features, valid_cat_features)
        else:
            st.error("X или y пусты перед train_test_split.")

        if ml_pipeline:
            st.subheader("Интерактивное предсказание")
            st.write("Введите данные для предсказания серьезности ДТП.")

            input_data = {}
            form_cols = st.columns(3) 
            feature_idx = 0
            
            # Используем X.columns (признаки ДО разделения) для создания полей ввода
            # Это гарантирует, что мы предлагаем пользователю все признаки, которые были в данных
            for feature_name in X.columns:
                current_col = form_cols[feature_idx % 3]
                with current_col:
                    # Проверяем принадлежность к valid_num_features / valid_cat_features, которые основаны на X_train
                    # Это означает, что если признак есть в X, но не попал в X_train (маловероятно, но возможно),
                    # для него не будет создан виджет. Это ОК, т.к. модель его не увидит.
                    if feature_name in numerical_features: # Используем исходные numerical_features для определения типа
                        if feature_name == 'Hour':
                            # Проверяем, что X[feature_name] не пуст перед вызовом median()
                            median_hour = int(X[feature_name].median()) if not X[feature_name].empty else 12
                            input_data[feature_name] = st.slider(f"{feature_name}", 0, 23, median_hour)
                        elif feature_name == 'Year' and X[feature_name].dtype == 'object': 
                            years_options = sorted(list(X[feature_name].unique())) # list() для безопасности
                            input_data[feature_name] = st.selectbox(f"{feature_name}", years_options, index=len(years_options)//2 if years_options else 0)
                        elif X[feature_name].dtype != 'object' and not X[feature_name].empty: 
                            min_v = float(X[feature_name].min())
                            max_v = float(X[feature_name].max())
                            med_v = float(X[feature_name].median())
                            input_data[feature_name] = st.number_input(f"{feature_name}", min_value=min_v, max_value=max_v, value=med_v, step= (max_v-min_v)/100 if max_v != min_v else 0.01) # step 0.01 если min=max
                        else: # Если признак числовой, но пустой (маловероятно после prepare_ml_data)
                            input_data[feature_name] = st.number_input(f"{feature_name} (данные отсутствуют)", value=0.0)
                    
                    elif feature_name in categorical_features: # Используем исходные categorical_features
                        unique_vals = sorted(list(X[feature_name].astype(str).unique()))
                        if not X[feature_name].empty and unique_vals:
                            default_val_cat = X[feature_name].mode()[0] if not X[feature_name].empty else unique_vals[0]
                            try:
                                default_idx_cat = unique_vals.index(str(default_val_cat))
                            except ValueError:
                                default_idx_cat = 0 if unique_vals else 0 # Добавил проверку unique_vals
                            input_data[feature_name] = st.selectbox(f"{feature_name}", unique_vals, index=default_idx_cat)
                        else:
                            input_data[feature_name] = st.text_input(f"{feature_name} (данные отсутствуют)", value="N/A")
                feature_idx += 1
            
            if st.button("Предсказать серьезность"):
                if not X_train.empty: # Убедимся, что X_train не пуст (модель обучена)
                    try:
                        # Создаем DataFrame с колонками в том же порядке, что и X_train
                        input_df_ordered = pd.DataFrame([input_data])[list(X_train.columns)]
                        
                        prediction_encoded = ml_pipeline.predict(input_df_ordered)
                        prediction_proba = ml_pipeline.predict_proba(input_df_ordered)
                        predicted_severity = label_encoder.inverse_transform(prediction_encoded)[0]
                        st.success(f"Предсказанная серьезность ДТП: **{predicted_severity}**")
                        st.subheader("Вероятности классов:")
                        proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
                        st.dataframe(proba_df)
                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
                        st.info(f"Входные данные (input_data): {input_data}")
                        st.info(f"Ожидаемые колонки (из X_train): {list(X_train.columns)}")
                        # st.info(f"DataFrame перед предсказанием (input_df_ordered): {input_df_ordered.to_dict()}") # Для детальной отладки
                else:
                    st.error("Модель не была обучена, так как обучающая выборка пуста.")


            st.subheader("Оценка модели (на тестовой выборке)")
            if not X_test.empty and not y_test.empty: # Проверяем y_test тоже
                y_pred_test = ml_pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_test)
                st.write(f"Точность модели: {accuracy:.4f}")
                # Убедимся, что label_encoder.classes_ не пуст
                target_names_report = label_encoder.classes_ if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0 else None
                report = classification_report(y_test, y_pred_test, target_names=target_names_report, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.warning("Тестовая выборка (X_test или y_test) пуста, оценка не производится.")
        # else: # Ошибка ml_data_prepared_successfully уже выведена


# --- 5. Раздел "Прогнозирование количества ДТП (Временные ряды)" ---
elif app_mode == "Прогнозирование количества ДТП (Временные ряды)":
    st.title("Прогнозирование количества ДТП (Временные ряды)")

    if df_main.empty:
        st.error("Данные не загружены. Невозможно продолжить с прогнозированием временных рядов.")
    else:
        monthly_accidents = get_time_series_data(df_main.copy())

        if monthly_accidents.empty:
            st.warning("Не удалось подготовить данные для временных рядов.")
        else:
            st.subheader("Ежемесячное количество ДТП")
            fig_ts = px.line(monthly_accidents, x=monthly_accidents.index, y='Num_Accidents',
                             labels={'Num_Accidents': 'Количество ДТП', 'Accident Date': 'Дата'},
                             title="Ежемесячное количество ДТП")
            fig_ts.update_layout(xaxis_title='Дата', yaxis_title='Количество ДТП')
            st.plotly_chart(fig_ts, use_container_width=True)

            train_data_ts = pd.Series(dtype='float64')
            test_data_ts = pd.Series(dtype='float64')

            if len(monthly_accidents) > 6: 
                num_test_points = min(3, len(monthly_accidents) // 4) 
                train_data_ts = monthly_accidents[:-num_test_points]
                test_data_ts = monthly_accidents[-num_test_points:]
            else:
                train_data_ts = monthly_accidents # Используем все, если данных мало
            
            sarima_model = None # Инициализируем
            if not train_data_ts.empty:
                with st.spinner("Обучение модели SARIMA... Это может занять некоторое время."):
                    sarima_model = get_trained_sarima_model(train_data_ts)

            if sarima_model:
                st.subheader("Сводка по модели SARIMA")
                from io import StringIO
                import sys
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                try:
                    sarima_model.summary()
                    summary_text = captured_output.getvalue()
                except AttributeError: 
                    summary_text = sarima_model.summary().as_text()
                finally:
                    sys.stdout = old_stdout
                st.text(summary_text)

                st.subheader("Прогноз количества ДТП")
                n_periods_future_forecast = st.slider("Количество месяцев для прогноза в будущее:", 1, 24, 6)
                
                total_forecast_periods_ts = (len(test_data_ts) if not test_data_ts.empty else 0) + n_periods_future_forecast
                
                if total_forecast_periods_ts > 0:
                    forecast_obj = sarima_model.predict(n_periods=total_forecast_periods_ts, return_conf_int=True)
                    forecast_values_ts = forecast_obj[0]
                    conf_int_ts = forecast_obj[1]
                    
                    if not test_data_ts.empty:
                        forecast_index_start_ts = test_data_ts.index[0]
                    elif not train_data_ts.empty:
                        forecast_index_start_ts = train_data_ts.index[-1] + pd.DateOffset(months=1)
                    else: 
                        forecast_index_start_ts = pd.to_datetime('today').normalize() + pd.DateOffset(months=1)
                        st.warning("Нет данных для определения начальной точки прогноза.")

                    forecast_idx_ts = pd.date_range(start=forecast_index_start_ts, periods=total_forecast_periods_ts, freq='M')
                    
                    forecast_series_ts = pd.Series(forecast_values_ts, index=forecast_idx_ts)
                    lower_bound_ts = pd.Series(conf_int_ts[:, 0], index=forecast_idx_ts)
                    upper_bound_ts = pd.Series(conf_int_ts[:, 1], index=forecast_idx_ts)

                    fig_forecast_plt = plt.figure(figsize=(12, 6))
                    ax_plt = fig_forecast_plt.gca()
                    monthly_accidents.plot(ax=ax_plt, label='Исторические данные')
                    if not test_data_ts.empty:
                         test_data_ts.plot(ax=ax_plt, label='Реальные значения (тест)', style='.-', marker='o')
                    forecast_series_ts.plot(ax=ax_plt, label='Прогноз SARIMA', style='--', marker='x')
                    ax_plt.fill_between(forecast_idx_ts, lower_bound_ts, upper_bound_ts, color='pink', alpha=0.3, label='Доверительный интервал 95%')
                    plt.title('Прогноз количества ДТП с помощью SARIMA')
                    plt.xlabel('Дата')
                    plt.ylabel('Количество ДТП')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(fig_forecast_plt)
                    
                    if not test_data_ts.empty:
                        st.write("Прогноз на тестовом периоде:")
                        actual_test_forecast_values = forecast_series_ts.iloc[:len(test_data_ts)]
                        comparison_df_ts = pd.DataFrame({'Реальные': test_data_ts, 'Прогноз': actual_test_forecast_values.round(2)})
                        st.dataframe(comparison_df_ts)
                        
                        mae_ts = mean_absolute_error(test_data_ts, actual_test_forecast_values)
                        rmse_ts = np.sqrt(mean_squared_error(test_data_ts, actual_test_forecast_values))
                        st.write(f"MAE на тестовых данных: {mae_ts:.2f}")
                        st.write(f"RMSE на тестовых данных: {rmse_ts:.2f}")
                else:
                    st.info("Нет периодов для прогноза (тестовые данные отсутствуют и прогноз на будущее 0 месяцев).")
            else:
                st.error("Модель SARIMA не была обучена (возможно, из-за недостатка данных).")
    
st.sidebar.markdown("---")
st.sidebar.info("Разработано на основе анализа данных ДТП.")
