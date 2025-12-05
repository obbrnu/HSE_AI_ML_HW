import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path

st.set_page_config(layout='wide')
st.title('AI_HW1_Regression_with_inference_pro_Bobrun_Alexander')

@st.cache_resource
def load_notebook_data():
    try:
        pickle_path = Path(__file__).parent / 'notebook_data.pickle'
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error('.pickle-файл не найден')
        return None

@st.cache_data
def extract_options_from(_notebook_data):
    if _notebook_data is None:
        return {}
    
    options = {}

    df_train = _notebook_data['df_train_processed']

    brand_columns = [col for col in _notebook_data['feature_names'] if col.startswith('name_')]
    brands = [col.replace('name_', '') for col in brand_columns]

    all_brands = list(df_train['name'].unique())
    options['brands'] = sorted(list(set(brands + all_brands)))

    fuel_columns = [col for col in _notebook_data['feature_names'] if col.startswith('fuel_')]
    fuels = [col.replace('fuel_', '') for col in fuel_columns]
    all_fuels = list(df_train['fuel'].unique())
    options['fuels'] = sorted(list(set(fuels + all_fuels)))

    seller_columns = [col for col in _notebook_data['feature_names'] if col.startswith('seller_type_')]
    sellers = [col.replace('seller_type_', '') for col in seller_columns]
    all_sellers = list(df_train['seller_type'].unique())
    options['sellers'] = sorted(list(set(sellers + all_sellers)))

    transmission_columns = [col for col in _notebook_data['feature_names'] if col.startswith('transmission_')]
    transmissions = [col.replace('transmission_', '') for col in transmission_columns]
    all_transmissions = list(df_train['transmission'].unique())
    options['transmissions'] = sorted(list(set(transmissions + all_transmissions)))

    owner_columns = [col for col in _notebook_data['feature_names'] if col.startswith('owner_')]
    owners = [col.replace('owner_', '') for col in owner_columns]
    all_owners = [owner for owner in df_train['owner'].unique() if owner != 'Test Drive Car']
    options['owners'] = sorted(list(set(owners + all_owners)))

    seats_columns = [col for col in _notebook_data['feature_names'] if col.startswith('seats_')]
    seats = [int(col.replace('seats_', '')) for col in seats_columns]
    all_seats = list(df_train['seats'].unique())
    options['seats'] = sorted(list(set(seats + all_seats)))
    
    options['year_range'] = (int(df_train['year'].min()), int(df_train['year'].max()))
    options['km_range'] = (0, int(df_train['km_driven'].max()))
    options['mileage_range'] = (float(df_train['mileage'].min()), float(df_train['mileage'].max()))
    options['engine_range'] = (int(df_train['engine'].min()), int(df_train['engine'].max()))
    options['max_power_range'] = (float(df_train['max_power'].min()), float(df_train['max_power'].max()))
    options['torque_range'] = (float(df_train['torque'].min()), float(df_train['torque'].max()))
    options['max_torque_rpm_range'] = (int(df_train['max_torque_rpm'].min()), int(df_train['max_torque_rpm'].max()))

    return options

notebook_data = load_notebook_data()

if notebook_data is None:
    st.stop()

df_train = notebook_data['df_train_processed']
options = extract_options_from(notebook_data)

st.sidebar.title('Навигация')
page = st.sidebar.selectbox('Текущий раздел', ['EDA', 'Информация о модели', 'Предсказание цен'])

if page == 'EDA':
    st.header('Исследовательский анализ данных (EDA)')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Количество записей', df_train.shape[0])
    with col2:
        st.metric('Количество признаков', df_train.shape[1])
    with col3:
        st.metric('Количество пропущенных значений', df_train.isnull().sum().sum())

    st.subheader('Распределение целевой переменной')

    fig_hist = px.histogram(df_train, x='selling_price', nbins=50, title='Распределение цен на автомобили')
    fig_hist.update_layout(xaxis_title='Цена', yaxis_title='Частота')
    st.plotly_chart(fig_hist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Основные статистики цен')
        price_stats = df_train['selling_price'].describe()
        st.dataframe(price_stats)
    
    with col2:
        st.subheader('Распределение цен')
        fig_box = px.box(df_train, y='selling_price')
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader('Корреляционная матрица числовых признаков')
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    corr_matrix = df_train[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect='auto',
                        title='Корреляционная матрица')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader('Распределения категориальных признаков')

    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

    for category in notebook_data['categorical_features']:
        fig = px.pie(df_train, names=category, title=f'Распределение: {category}')
        st.plotly_chart(fig, use_container_width=True)

    for feature in notebook_data['numeric_features']:
        st.subheader(f'Зависимость цены от {feature}')
        fig = px.scatter(df_train, x=feature, y='selling_price')
        fig.update_layout(xaxis_title=feature, yaxis_title='selling_price')
        st.plotly_chart(fig, use_container_width=True)

elif page == 'Информация о модели':
    st.header('Информация о модели')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Тип модели', 'Ridge Regression')
    with col2:
        st.metric('Количество признаков', len(notebook_data['feature_names']))
    with col3:
        st.metric('Параметр регуляризации (alpha)', notebook_data['best_model'].alpha)
    with col4:
        st.metric('Fit intercept:', notebook_data['best_model'].fit_intercept)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('train_R2', notebook_data['best_model_train_R2'])
    with col2:
        st.metric('test_R2', notebook_data['best_model_test_R2'])
    with col3:
        st.metric('train_MSE', notebook_data['best_model_train_MSE'])
    with col4:
        st.metric('test_MSE', notebook_data['best_model_test_MSE'])

    st.subheader('Веса признаков модели')

    feature_names = notebook_data['feature_names']
    coefficients = notebook_data['best_model'].coef_

    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefficients,
        'Abs_Weight': np.abs(coefficients)
    }).sort_values('Abs_Weight', ascending=False)

    top_features = weights_df

    fig_weights = px.bar(top_features,
                        x='Weight',
                        y='Feature',
                        orientation='h',
                        color='Weight',
                        color_continuous_scale='RdBu')

    fig_weights.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_weights, use_container_width=True)

    st.dataframe(weights_df, use_container_width=True)

    st.subheader('Распределение весов')
    fig_dist = px.histogram(weights_df, x='Weight', nbins=30)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader('Интерпретация модели')

    col1, col2 = st.columns(2)
    with col1:
        st.write('Признаки, максимально увеличивающие цену:')
        for _, row in weights_df[weights_df['Weight'] > 0].head(5).iterrows():
            st.write(f'{row["Feature"]}: {row["Weight"]:.2f}')
    with col2:
        st.write('Признаки, максимально уменьшающие цену:')
        for _, row in weights_df[weights_df['Weight'] < 0].head(5).iterrows():
            st.write(f'{row["Feature"]}: {row["Weight"]:.2f}')

elif page == 'Предсказание цен':
    st.header('Предсказание цен')

    input_method = st.radio('Cпособ ввода данных:', ['Ввод', 'Готовый CSV файл'])

    if input_method == 'Ввод':
        st.subheader('Введите характеристики автомобиля:')

        col1, col2 = st.columns(2)
        
        with col1:
            name = st.selectbox('name', options=options['brands'])

            year = st.slider('year', min_value=options['year_range'][0], max_value=options['year_range'][1], value=2000)

            km_driven = st.number_input('km_driven', min_value=0, max_value=options['km_range'][1], value=50000)

            fuel = st.selectbox('fuel', options=options['fuels'])

            seller_type = st.selectbox('seller_type', options=options['sellers'])

            transmission = st.selectbox('transmissions', options=options['transmissions'])

            owner = st.selectbox('owner', options=options['owners'])

        with col2:
            
            mileage = st.number_input('mileage',
                                    min_value=options['mileage_range'][0],
                                    max_value=options['mileage_range'][1], 
                                    value=float((options['mileage_range'][0] + options['mileage_range'][1]) / 2))
            
            engine = st.number_input('engine',
                                   min_value=options['engine_range'][0],
                                   max_value=options['engine_range'][1], 
                                   value=2000)

            max_power = st.number_input('max_power',
                                      min_value=options['max_power_range'][0],
                                      max_value=options['max_power_range'][1],
                                      value=200.0)

            torque = st.number_input('torque',
                                   min_value=options['torque_range'][0],
                                   max_value=options['torque_range'][1], 
                                   value=2000.0)

            seats = st.selectbox('seats', options=options['seats'])

            max_torque_rpm = st.number_input('max_torque_rpm',
                                           min_value=options['max_torque_rpm_range'][0],
                                           max_value=options['max_torque_rpm_range'][1],
                                           value=10000)

        input_data = pd.DataFrame({
            'name': [name],
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'torque': [torque],
            'seats': [seats],
            'max_torque_rpm': [max_torque_rpm]
        })
            
        try:
            input_data['name'] = input_data['name'].str.split(' ').str[0]

            numeric_features = notebook_data['numeric_features']
            X_numeric = input_data[numeric_features]

            categorical_prefixes = [f'{cat}_' for cat in notebook_data['categorical_features']]
            categorical_columns = [col for col in notebook_data['feature_names']
                                  if any(col.startswith(prefix) for prefix in categorical_prefixes)]

            X_cat_encoded = pd.DataFrame(0, index=[0], columns=categorical_columns, dtype=bool)

            if f'name_{name}' in categorical_columns:
                X_cat_encoded[f'name_{name}'] = True

            if f'fuel_{fuel}' in categorical_columns:
                X_cat_encoded[f'fuel_{fuel}'] = True

            if f'seller_type_{seller_type}' in categorical_columns:
                X_cat_encoded[f'seller_type_{seller_type}'] = True

            if f'transmission_{transmission}' in categorical_columns:
                X_cat_encoded[f'transmission_{transmission}'] = True

            if f'owner_{owner}' in categorical_columns:
                X_cat_encoded[f'owner_{owner}'] = True

            if f'seats_{seats}' in categorical_columns:
                X_cat_encoded[f'seats_{seats}'] = True

            X_numeric_scaled = notebook_data['scaler'].transform(X_numeric)
            X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

            X_combined = pd.concat(
              [
                X_numeric_scaled_df.reset_index(drop=True),
                X_cat_encoded.reset_index(drop=True)
              ],
              axis=1,
            )

            for col in notebook_data['feature_names']:
                if col not in X_combined.columns:
                    X_combined[col] = False

            X_final = X_combined.reindex(columns=notebook_data['feature_names'], fill_value=False)

            prediction = notebook_data['best_model'].predict(X_final)[0]

            st.success(f'Предсказанная цена: {prediction:,.0f}')

        except Exception as e:
            st.error(f'Error: {str(e)}')

    else:
        st.subheader('Загрузите CSV файл с данными:')

        example_df = pd.DataFrame({
            'name': ['Maruti', 'Honda', 'BMW'],
            'year': [2015, 2018, 2020],
            'km_driven': [50000, 30000, 15000],
            'fuel': ['Petrol', 'Petrol', 'Diesel'],
            'seller_type': ['Individual', 'Dealer', 'Dealer'],
            'transmission': ['Manual', 'Manual', 'Automatic'],
            'owner': ['First Owner', 'First Owner', 'Second Owner'],
            'mileage': [15.0, 18.0, 16.5],
            'engine': [1200, 1500, 2000],
            'max_power': [82, 120, 190],
            'torque': [115, 145, 400],
            'seats': [5, 5, 5],
            'max_torque_rpm': [4000, 3600, 1750],
        })

        required_columns = example_df.columns

        with st.expander('Требуемый формат CSV файла'):
            st.write('Список обязательных полей:')
            st.dataframe(required_columns)

            st.write('Пример(можно скачать):')
            st.dataframe(example_df)
        
        uploaded_file = st.file_uploader('Выберите CSV файл', type=['csv'])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write('Выбранный CSV файл:')
                st.dataframe(input_df.head())
                
                if st.button('Сделать предсказание'):
                    try:
                        required_columns = example_df.columns

                        missing_columns = [col for col in required_columns if col not in input_df.columns]
                        
                        if missing_columns:
                            st.error(f'Отсутствуют обязательные колонки: {missing_columns}')
                        else:
                            df_to_predict = input_df.copy()

                            df_to_predict['name'] = df_to_predict['name'].str.split(' ').str[0]

                            if len(df_to_predict) > 0:
                                predictions = []
                                errors = []
                                
                                for idx, row in df_to_predict.iterrows():
                                    try:
                                        numeric_features = notebook_data['numeric_features']
                                        X_numeric = pd.DataFrame([row[numeric_features].values], columns=numeric_features)

                                        categorical_prefixes = [f'{cat}_' for cat in notebook_data['categorical_features']]
                                        categorical_columns = [col for col in notebook_data['feature_names']
                                                             if any(col.startswith(prefix) for prefix in categorical_prefixes)]
                                        
                                        X_cat_encoded = pd.DataFrame(0, index=[0], columns=categorical_columns, dtype=bool)

                                        for cat_feature in notebook_data['categorical_features']:
                                            value = row[cat_feature]
                                            col_name = f'{cat_feature}_{value}'
                                            if col_name in categorical_columns:
                                                X_cat_encoded[col_name] = True

                                        X_numeric_scaled = notebook_data['scaler'].transform(X_numeric)
                                        X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

                                        X_combined = pd.concat(
                                          [
                                            X_numeric_scaled_df.reset_index(drop=True),
                                            X_cat_encoded.reset_index(drop=True),
                                          ],
                                          axis=1,
                                        )

                                        for col in notebook_data['feature_names']:
                                            if col not in X_combined.columns:
                                                X_combined[col] = False
                                        
                                        X_final = X_combined.reindex(columns=notebook_data['feature_names'], fill_value=False)

                                        prediction = notebook_data['best_model'].predict(X_final)[0]
                                        predictions.append(prediction)
                                        
                                    except Exception as e:
                                        st.error(f'Error: {str(e)}')

                                result_df = df_to_predict.copy()
                                result_df['predicted_price'] = predictions

                                st.subheader('Результаты')

                                display_df = result_df
                                
                                st.dataframe(display_df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f'Error: {str(e)}')

            except Exception as e:
                st.error(f'Error: {str(e)}')
