"""
Tests for database/database.py
Uses unittest.mock to avoid requiring a real MySQL connection.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from types import SimpleNamespace


class TestDataBase:
    """Tests for the DataBase class with mocked MySQL connections."""

    @staticmethod
    def _make_config():
        return SimpleNamespace(
            connection={
                "user": "test",
                "password": "test",
                "host": "localhost",
                "raise_on_warnings": True,
            },
            database_name="test_db",
        )

    @patch("database.database.mysql.connector.connect")
    def test_init_connects(self, mock_connect):
        mock_connect.return_value = MagicMock(autocommit=False)
        from database.database import DataBase
        cfg = self._make_config()
        db = DataBase(cfg)
        mock_connect.assert_called_once_with(**cfg.connection)
        assert db.database_name == "test_db"

    @patch("database.database.mysql.connector.connect")
    def test_use_database(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        db.use_database()

        mock_cursor.execute.assert_called_with("USE test_db")

    @patch("database.database.mysql.connector.connect")
    def test_create_table(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        db.create_table(["edge1", "edge2"])

        # Should have called execute for each edge id
        assert mock_cursor.execute.call_count >= 2

    @patch("database.database.mysql.connector.connect")
    def test_insert_data(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        data = (1, 1000.0, 1001.0, "result", "log")
        db.insert_data("edge1", data)

        mock_cursor.execute.assert_called_once()
        mock_cnx.commit.assert_called_once()

    @patch("database.database.mysql.connector.connect")
    def test_select_result(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(1, 100.0, 101.0, "res", "log")]
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        results = db.select_result("edge1")

        assert results is not None
        assert len(results) == 1

    @patch("database.database.mysql.connector.connect")
    def test_select_one_result(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, 100.0, 101.0, "res", "log")
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        result = db.select_one_result("edge1", 1)

        assert result is not None
        assert result[0] == 1

    @patch("database.database.mysql.connector.connect")
    def test_update_data(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        data = (1001.0, "updated_result", "updated_log", 1)
        db.update_data("edge1", data)

        mock_cursor.execute.assert_called_once()
        mock_cnx.commit.assert_called_once()

    @patch("database.database.mysql.connector.connect")
    def test_clear_table(self, mock_connect):
        mock_cnx = MagicMock(autocommit=False)
        mock_cursor = MagicMock()
        mock_cnx.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_cnx

        from database.database import DataBase
        db = DataBase(self._make_config())
        db.clear_table("edge1")

        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0][0]
        assert "truncate" in call_args.lower()

    @patch("database.database.mysql.connector.connect")
    def test_sql_templates_contain_placeholders(self, mock_connect):
        mock_connect.return_value = MagicMock(autocommit=False)
        from database.database import DataBase
        db = DataBase(self._make_config())

        # Verify SQL templates can be formatted with table names
        assert "{}" in db.table_desc
        assert "{}" in db.insert_desc
        assert "{}" in db.select_desc
        assert "{}" in db.update_desc
