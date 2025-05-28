
import os
import re

def update_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace st.session_state.access_token with session-specific key
        # Use regex to match exact attribute access
        updated_content = re.sub(
            r'st\.session_state\.access_token\b',
            r'st.session_state[f\'access_token_{st.session_state.session_id}\']',
            content
        )

        # Write back if changed
        if updated_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"Updated {file_path}")
        else:
            print(f"No changes needed in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # List of Python files to process
    files = [
        'DailyZenfin.py',
        'DailyMotivation.py',
        'mule.py',
        'test.py',
        'app.py'
    ]

    # Ensure session_id initialization in each file
    init_code = """
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if f'access_token_{st.session_state.session_id}' not in st.session_state:
    st.session_state[f'access_token_{st.session_state.session_id}'] = None
"""

    for file_path in files:
        if os.path.exists(file_path):
            update_file(file_path)
            # Add session_id initialization if not present
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'session_id' not in content:
                with open(file_path, 'r+', encoding='utf-8') as f:
                    content = f.read()
                    # Insert after imports
                    lines = content.split('\n')
                    import_end = 0
                    for i, line in enumerate(lines):
                        if not line.startswith('import') and not line.startswith('from'):
                            import_end = i
                            break
                    lines.insert(import_end, init_code)
                    f.seek(0)
                    f.write('\n'.join(lines))
                    print(f"Added session_id initialization to {file_path}")
        else:
            print(f"File not found: {file_path}")

if __name__ == '__main__':
    main()
