import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
RAW_DIR = Path('raw')
CLEANED_DIR = Path('cleaned')

BASIC_FILE = 'basic'
INTAKE_FORM_FILE = 'intake_form'
OVERALL_FILE = 'grades'
FINAL_ASSESSMENT_FILE = 'final_assessment'
STATUS_FILE = 'status'


def read_csv(filename: str) -> pd.DataFrame | None:
    dir_path = DATA_DIR / RAW_DIR

    return pd.read_csv(dir_path / f'{filename}.csv')


def write_csv(df: pd.DataFrame, filename: str):
    dir_path = DATA_DIR / CLEANED_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(dir_path / f'{filename}.csv', index=False)

    return filename


INTAKE_FORM = {
    'col_name': [
        'id', 'timestamp', 'consent_given',
        'recruitment_source',
        'dsc_affiliation', 'math_affiliation',
        'class_standing', 'stats_courses_taken',
        'stats_confidence', 'chebyshev_familiarity', 'python_skill_level'
    ],
    'dsc_affiliation_map': {
        'I am a Data Science major': 'major',
        'I am a Data Science minor': 'minor',
        'Neither of the above': 'neither'
    },
    'math_affiliation_map': {
        'I am a Math major': 'major',
        'I am a Math minor': 'minor',
        'Neither of the above': 'neither'
    },
    'manual_cleaning': {
        'class_standing': {
            20: 'Third year (first year transfer)',
            28: 'Second year',
            29: 'Third year (first year transfer)'
        },
        'stats_courses_taken': {
            28: 'None of the above'
        }
    },
    'check_format': {
        'recruitment_source': (
            r'^(?:DSC|CSE) \d{2,3}'
        ),
        'class_standing': (
            r'^(?:First|Second|Third|Fourth) year'
            r'(?: \((?:first|second) year transfer\))?$'
        )
    }

}

OVERALL = {
    'col_keep': [
        'id',
        'Chebyshev Coding Exercises',
        'Chebyshev Coding Exercises - Max Points',
        'Chebyshev Handwritten Exercises',
        'Chebyshev Handwritten Exercises - Max Points'
    ],
    'col_name': [
        'id',
        'coding_score', 'coding_total',
        'handwritten_score', 'handwritten_total'
    ]
}

NONE_GROUP = 1
CODING_GROUP = 2
HANDWRITTEN_GROUP = 3
BOTH_GROUP = 4


def clean_form() -> pd.DataFrame:
    df = read_csv(INTAKE_FORM_FILE)

    df.columns = INTAKE_FORM['col_name']
    df = df.drop(columns=['timestamp', 'consent_given'])

    df['dsc_affiliation'] = (
        df['dsc_affiliation']
        .map(INTAKE_FORM['dsc_affiliation_map'])
    )
    df['math_affiliation'] = (
        df['math_affiliation']
        .map(INTAKE_FORM['math_affiliation_map'])
    )

    for col_name, cleaning_dict in INTAKE_FORM['manual_cleaning'].items():
        for sid, v in cleaning_dict.items():
            df.loc[sid, col_name] = v

    for col_name, col_format in INTAKE_FORM['check_format'].items():
        class_standing_check = df[
            ~df[col_name].str.contains(col_format)
        ]
        if class_standing_check.shape[0] != 0:
            print('-' * 50)
            print(f'Need check {col_name} column')
            print(class_standing_check[['id', col_name]])
            print('-' * 50)

    return df


def clean_overall() -> pd.DataFrame:
    df = read_csv(OVERALL_FILE)
    df = df[OVERALL['col_keep']]

    df.columns = OVERALL['col_name']

    df['coding_score'] = df['coding_score'] / df['coding_total']
    df['handwritten_score'] = df['handwritten_score'] / df['handwritten_total']

    df = df.drop(columns=['coding_total', 'handwritten_total'])

    return df


def clean_final() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv(FINAL_ASSESSMENT_FILE)

    ids = df['id']
    df = df[[
        col for col in df.columns
        if re.match(r'^Question \d+\.\d+ (Score|Weight|Response)$', col)
    ]]

    dfs = []
    for q_num in df.columns.to_series().str.split(' ').str[1].unique():
        int_part, float_part = q_num.split('.')
        if int(float_part) % 2 == 0:
            continue
        new_q_num = f'{int_part}.{int(float_part) // 2 + 1}'

        dfs.append(pd.DataFrame({
            f'{new_q_num}_score': (
                df[f'Question {q_num} Score']
                .replace({0.1: 0})
            ),
            f'{new_q_num}_pts': (
                df[f'Question {q_num} Weight']
            ),
            f'{new_q_num}_confident_level': (
                df[
                    f'Question {int_part}.{int(float_part) + 1} Response'
                ].replace(
                    {
                        r'^5 - .+': 5,
                        r'^0 - .+': 0
                    }, regex=True
                ).astype(float)
            )
        }))
    df = pd.concat(dfs, axis=1)

    dfs = [ids]
    total_score = pd.Series(np.zeros(df.shape[0]))
    total_score_adj = pd.Series(np.zeros(df.shape[0]))
    total_points = pd.Series(np.zeros(df.shape[0]))
    for q_num in df.columns.to_series().str.split('_').str[0].unique():
        total_score += df[f'{q_num}_score']
        total_score_adj += (
                df[f'{q_num}_score']
                * df[f'{q_num}_confident_level'].fillna(3.5) / 5
        )
        total_points += df[f'{q_num}_pts']

        dfs.append(pd.DataFrame({
            f'{q_num}_score': df[f'{q_num}_score'] / df[f'{q_num}_pts'],
            f'{q_num}_confident_level': df[f'{q_num}_confident_level']
        }))
    df = pd.concat(dfs, axis=1)

    df_overall_right_part = pd.DataFrame({
        'id': ids,
        'final_score': total_score / total_points,
        'final_score_adj': total_score_adj / total_points
    })

    return df, df_overall_right_part


def clean_basic() -> pd.DataFrame:
    df = read_csv(BASIC_FILE)
    df.columns = ['id', 'section']

    return df


def completeness(df):
    coding_done = pd.notna(df['coding_score'])
    handwritten_done = pd.notna(df['handwritten_score'])
    final_done = pd.notna(df['final_score'])

    df['completed'] = (
                              (df['section'] == NONE_GROUP) & final_done
                              & ~coding_done & ~handwritten_done
                      ) | (
                              (df['section'] == CODING_GROUP) & final_done
                              & coding_done & ~handwritten_done
                      ) | (
                              (df['section'] == HANDWRITTEN_GROUP) & final_done
                              & handwritten_done & ~coding_done
                      ) | (
                              (df['section'] == BOTH_GROUP) & final_done
                              & coding_done & handwritten_done
                      )

    return df[['id', 'section', 'completed']]


if __name__ == '__main__':
    df_form = clean_form()
    df_overall_left = clean_overall()
    df_final, df_overall_right = clean_final()

    df_overall = df_overall_left.merge(df_overall_right, on='id')

    df_basic = clean_basic()

    df_completed = completeness(df_basic.merge(df_overall, on='id'))
    df_ids = df_completed[df_completed['completed']]['id'].to_frame()

    df_basic = df_ids.merge(df_basic, on='id', how='left')
    df_form = df_ids.merge(df_form, on='id', how='left')
    df_overall = df_ids.merge(df_overall, on='id', how='left')

    df_final = df_ids.merge(df_final, on='id', how='left')

    write_csv(df_completed, STATUS_FILE)
    write_csv(df_basic, BASIC_FILE)
    write_csv(df_form, INTAKE_FORM_FILE)
    write_csv(df_overall, OVERALL_FILE)
    write_csv(df_final, FINAL_ASSESSMENT_FILE)
    print('Generating Done')
