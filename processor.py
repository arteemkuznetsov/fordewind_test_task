from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd


class Processor:
    """Data normalizer."""

    _calc_columns: dict[str, Callable]

    def __init__(self):
        """Constructs all the necessary attributes for the data processor."""
        self._calc_columns = {
            'Прибыль (убыток) от продажи, RUB': self._calc_profit,
            'Возраст компании, years': self._get_age
        }

    def _calc_profit(self, data: pd.DataFrame, _id: int, year: int) -> Optional[np.float64]:
        """
        Calculate profit by formula.

        :param data: Input dataframe.
        :type data: pd.DataFrame
        :param _id: Row index.
        :type _id: int
        :param year: Column prefix.
        :type year: int
        :return: Calculated profit.
        :rtype: np.float64 or np.NaN
        """
        try:
            revenue = np.float64(data.loc[_id, f'{year}, Выручка, RUB'])
            cost_sales = np.float64(data.loc[_id, f'{year}, Себестоимость продаж, RUB'])
            adm_expenses = np.float64(data.loc[_id, f'{year}, Управленческие расходы, RUB'])
            com_expenses = np.float64(data.loc[_id, f'{year}, Коммерческие расходы, RUB'])
            return revenue - cost_sales - adm_expenses - com_expenses
        except KeyError:
            return np.nan

    def _get_age(self, data: pd.DataFrame, _id: int, year: int) -> Optional[int]:
        """
        Calculate company age.

        :param data: Input dataframe.
        :type data: pd.DataFrame
        :param _id: Row index.
        :type _id: int
        :param year: Column prefix.
        :type year: int
        :return: Calculated age.
        :rtype: int or np.NaN
        """
        age = year - data.loc[_id, 'Дата регистрации'].year
        return age if age > 0 else np.nan

    def _produce_column_name(self, request: dict) -> str:
        """
        Produce column name with prefix and postfix.

        :param request: Request from application dataframe.
        :type request: dict
        :return: Produced column name.
        :rtype: str
        """
        column_name = request['field_name']
        additional = ''
        if 'prev' in request and request['prev'] != 0:
            additional += f' {str("Prev" if request["prev"] > 0 else "Next") * abs(request["prev"])}'
        if 'last_available' in request:
            additional += f' LA{request["last_available"]}'
        return column_name + additional

    def _field_not_empty(self, data: pd.DataFrame, field_name: str, idx: int) -> bool:
        """
        Check if field exists and contains anything but np.NaN.

        :param data: Input dataframe.
        :type data: pd.DataFrame
        :param field_name: Complete column name.
        :type field_name: str
        :param idx: Index of row.
        :type idx: int
        :return: Is field not empty.
        :rtype: bool
        """
        return field_name in data.loc[idx] and pd.notna(data.loc[idx, field_name])

    def get_data(
            self,
            data: pd.DataFrame,
            applications: pd.DataFrame,
            request: list[dict]
    ) -> pd.DataFrame:
        """
        Normalize data according to scheme (application dataframe and request dict).

        :param data: Input dataframe.
        :type data: pd.DataFrame
        :param applications: Applications dataframe.
        :type applications: pd.DataFrame.
        :param request: scheme of future columns
        :type request: list
        :return: Normalized dataframe.
        :rtype: pd.DataFrame
        """
        df_out = pd.DataFrame()
        for k, row_in in applications.iterrows():
            row_out = {}
            for req in request:
                column_name = self._produce_column_name(req)
                year_to_select = row_in['year'] - req['prev'] if 'prev' in req else row_in['year']
                if req['field_name'] in self._calc_columns:
                    column_value = self._calc_columns[req['field_name']](data, row_in['_id'], year_to_select)
                else:
                    field_to_select = f'{year_to_select}, {req["field_name"]}'
                    if self._field_not_empty(data, field_to_select, row_in['_id']):
                        column_value = data.loc[row_in['_id'], field_to_select]
                    else:
                        column_value = np.nan
                        if 'last_available' in req:
                            for i in range(year_to_select - 1, year_to_select - req['last_available'] - 1, -1):
                                field_to_select = f'{i}, {req["field_name"]}'
                                if self._field_not_empty(data, field_to_select, row_in['_id']):
                                    column_value = data.loc[row_in['_id'], field_to_select]
                                    break
                row_out[column_name] = column_value
            df_out = pd.concat([df_out, pd.DataFrame([row_out])], ignore_index=True)
        return df_out
