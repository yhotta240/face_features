import os

def list_files_in_scripts_folder():
    scripts_folder = 'scripts'  # scriptsフォルダのパス
    files = os.listdir(scripts_folder)
    return files

def run_selected_file(selected_file):
    scripts_folder = 'scripts'  # scriptsフォルダのパス
    file_path = os.path.join(scripts_folder, selected_file)
    os.system(f'python {file_path}')  # 選択されたファイルを実行

def main():
    files = list_files_in_scripts_folder()
    print("使用可能なファイル: ")
    for index, file in enumerate(files, start=1):
        print(f"{index}. {file}")

    selection = input("ファイル番号を入力してください: ")
    try:
        selection_index = int(selection) - 1
        if 0 <= selection_index < len(files):
            selected_file = files[selection_index]
            run_selected_file(selected_file)
        else:
            print("無効な選択です。正しい番号を入力してください。")
    except ValueError:
        print("無効な入力です。番号を入力してください。")

if __name__ == "__main__":
    main()
