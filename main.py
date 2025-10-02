"""
Main module with GUI
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
import os
from YT_data import *
from YT_bot import *
import functools
import threading

def clear_layout(layout):
    """
    Util function to fully delete any layout
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        else:
            clear_layout(item.layout())

class AddDataTab(QWidget):
    def __init__(self, options):
        super().__init__()
        self.initUI(options)

    def initUI(self, options):
        tab_layout = QVBoxLayout()
        tab_layout.setAlignment(Qt.AlignTop)

        ### Input area ###
        self.input_area = QVBoxLayout()

        # Table select level
        table_select = QHBoxLayout()
        label = QLabel("Select table:")
        label.setFixedWidth(100)
        table_select.addWidget(label)
        self.comboBox = QComboBox()
        self.comboBox.currentIndexChanged.connect(self.load_columns_for_input)

        for option in options:
            self.comboBox.addItem(option)
        table_select.addWidget(self.comboBox)

        # Add layouts to the tab
        tab_layout.addLayout(table_select)
        tab_layout.addLayout(self.input_area)
        tab_layout.addStretch()

        ### Button to save data and clear input fields ###
        action_area = QHBoxLayout()
        action_area.addStretch()
        save_button = QPushButton('Save', self)
        save_button.setFixedWidth(100)
        action_area.addWidget(save_button)
        save_button.clicked.connect(self.save_data)
        tab_layout.addLayout(action_area)

        self.setLayout(tab_layout)

    def get_selected_option(self):
        return self.comboBox.currentText()

    def load_columns_for_input(self):
        # Clear existing input fields
        clear_layout(self.input_area)

        ### Create new input area ###
        # Get selected table name and load its columns
        table_name = self.get_selected_option()
        data, columns = get_table_data(table_name)

        # Define columns to exclude
        exclude_columns = ["stock_video_id",\
                           "stock_music_id"]

        self.inputs = {}
        # Cycle trough all columns in the selected table
        for column in columns:
            if column in exclude_columns:
                continue
            row_layout = QHBoxLayout()
            label = QLabel(column)
            label.setFixedWidth(100)
            input_field = QLineEdit()
            row_layout.addWidget(label)
            row_layout.addWidget(input_field)
            self.input_area.addLayout(row_layout)
            self.inputs[column] = input_field

    def save_data(self):
        # Create data dict
        table = self.get_selected_option()
        data = {}
        for column, input_field in self.inputs.items():
            data[column] = input_field.text()

        # Try to insert the data
        try:
            if data["name"] == '':
                raise ValueError(f'Name is not filled in')
            # Look for the file in the appropriate table and try to find the file
            files = os.listdir(table)
            matching_file = [file for file in files if file.startswith(data['name'])]

            # In case there are files with the same name with different file extensions
            if len(matching_file) > 1:
                raise ValueError(f'Multiple files found with the name:\n{data["name"]}')
            elif len(matching_file) == 1:
                # Adds the file extension
                data['name'] = matching_file[0]
                insert_data(table, **data)
            else:
                raise FileNotFoundError(f"File was not found:\n{data['name']}")

        # Diplay any errors that may occur
        except Exception as e:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(f"An error has occurred while inserting the data:\n{e}")
            msg_box.setWindowTitle("Error")
            msg_box.exec_()

        # If insertion successful -> load the input field again
        else:
            self.load_columns_for_input()


class ViewDataTab(QWidget):
    def __init__(self, options):
        super().__init__()
        self.initUI(options)

    def initUI(self, options):
        tab_layout = QVBoxLayout()

        ### Table select level ###
        table_select = QHBoxLayout()

        # Add label
        label = QLabel("Select table:")
        label.setFixedWidth(100)
        table_select.addWidget(label)

        # Add comboBox
        self.comboBox = QComboBox()
        for option in options:
            self.comboBox.addItem(option)
        self.comboBox.currentIndexChanged.connect(self.display_table_data)
        table_select.addWidget(self.comboBox)

        tab_layout.addLayout(table_select)

        ### Table level ###
        self.table_widget = QTableWidget()
        self.table_widget.verticalHeader().setVisible(False)
        tab_layout.addWidget(self.table_widget)

        self.setLayout(tab_layout)

    def get_selected_option(self):
        return self.comboBox.currentText()

    def display_table_data(self):
        data, columns = get_table_data(self.get_selected_option())
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels(columns)
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))


class YouTubeBotTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """ Tab Layout"""
        self.tab_layout = QVBoxLayout()
        self.bot = YouTubeBOT()

        ### Header ###
        self.header_groupbox = QGroupBox()
        self.header_groupbox.setTitle('YouTubeBOT overview')
        self.header_layout = QVBoxLayout()
        self.header_layout.setAlignment(Qt.AlignTop)

        # Number of channels
        num_of_channels_layout = QHBoxLayout()
        label = QLabel("Number of channels:")
        label.setFixedWidth(100)
        num_of_channels_layout.addWidget(label)
        self.num_of_channels_display = QLabel(str(len(self.bot.channels)))
        num_of_channels_layout.addWidget(self.num_of_channels_display)

        # Importing status
        importing_layout = QHBoxLayout()
        label = QLabel("Importing:")
        label.setFixedWidth(100)
        importing_layout.addWidget(label)
        self.importing_checkbox = QCheckBox()
        if self.bot.importing:  # Attr is currently set at True
            self.importing_checkbox.setChecked(True)
        self.importing_checkbox.stateChanged.connect(functools.partial(self.change_attribute, self.bot, 'importing'))
        importing_layout.addWidget(self.importing_checkbox)

        # Add header
        self.header_layout.addLayout(num_of_channels_layout)
        self.header_layout.addLayout(importing_layout)
        self.header_groupbox.setLayout(self.header_layout)
        self.tab_layout.addWidget(self.header_groupbox)

        ### Channels ###
        self.channel_panel = QVBoxLayout()

        self.update_info()

        self.tab_layout.addLayout(self.channel_panel)

        self.tab_layout.addStretch()
        self.setLayout(self.tab_layout)

        ### Updating ###
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_info)
        self.timer.start(1000)  # milliseconds

    def change_attribute(self, channel, attribute, value):
        setattr(channel, attribute, bool(value))

    def create_channel_groupbox(self, channel):
        # Get attr of channel
        channel_dict = channel.__dict__

        # Dict of diplayed attr - key: attr, value: toggable?
        displayed_attributes = {
            'theme' : False,
            'status' : False,
            'buffer' : False,
            'creating_videos' : True,
            'uploading' : True
            }

        # Prepare groupbox
        channel_groupbox = QGroupBox()
        channel_groupbox.setTitle(str(channel_dict['channel_name']))
        channel_layout = QVBoxLayout()
        channel_layout.setAlignment(Qt.AlignTop)

        # Add rows with attrs
        for attribute in displayed_attributes.keys():
            # Prepare row
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignLeft)
            attr_label = QLabel(str(attribute) + ':')
            attr_label.setFixedWidth(100)
            row.addWidget(attr_label)

            # Add value
            if attribute == 'buffer':
                value = str(len(channel_dict[attribute]))
            else:
                value = str(channel_dict[attribute])

            # Add Checkbox
            if displayed_attributes[attribute]: # Attr is toggable
                checkbox = QCheckBox()

                # Set check box to match current state
                if channel_dict[attribute]:
                    checkbox.setChecked(True)

                # Set uncheckable if no 'client_secrets.json'
                if not os.path.exists(os.path.join('Channels', channel.channel_name, 'client_secrets.json')):
                    checkbox.setCheckable(False)

                checkbox.stateChanged.connect(functools.partial(self.change_attribute, channel, attribute))
                row.addWidget(checkbox)
            else:
                value_label = QLabel(value)
                row.addWidget(value_label)

            # Add row
            channel_layout.addLayout(row)

        channel_groupbox.setLayout(channel_layout)

        return channel_groupbox

    def update_info(self):
        # Update overveiw
        self.num_of_channels_display.setText(str(len(self.bot.channels)))
        self.importing_checkbox.setChecked(self.bot.importing)

        # Update channel panel
        clear_layout(self.channel_panel)
        for channel in self.bot.channels:
            channel_groupbox = self.create_channel_groupbox(channel)
            self.channel_panel.addWidget(channel_groupbox)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create a window
        self.setWindowTitle('YouTubeBOT')
        self.setGeometry(100, 100, 600, 400)

        # Create tab widget
        self.tabs = QTabWidget()
        self.video_creation_tab = YouTubeBotTab()
        self.add_media_tab = AddDataTab(["stock_video", "stock_music"])
        self.data_view_tab = ViewDataTab(["stock_video", "stock_music", "post", "video", "video_stock_video", "video_stock_music"])

        # Add tabs to tab widget
        self.tabs.addTab(self.video_creation_tab, 'Video Creation')
        self.tabs.addTab(self.add_media_tab, 'Add Stock Media')
        self.tabs.addTab(self.data_view_tab, 'View Tables')

        # Set central widget
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        # Start cleanup in a separate thread
        cleanup_thread = threading.Thread(target= self.video_creation_tab.bot.cleanup)
        cleanup_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

