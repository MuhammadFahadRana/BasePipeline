SET NOCOUNT ON;

DECLARE @DatabaseName SYSNAME = N'VideoSemanticDB';

IF DB_ID(@DatabaseName) IS NOT NULL
BEGIN
    PRINT 'Dropping existing database [' + @DatabaseName + ']...';

    DECLARE @DropSql NVARCHAR(MAX) =
        N'ALTER DATABASE [' + REPLACE(@DatabaseName, ']', ']]') + N'] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;' +
        N' DROP DATABASE [' + REPLACE(@DatabaseName, ']', ']]') + N'];';
    EXEC (@DropSql);
END
ELSE
BEGIN
    PRINT 'Database [' + @DatabaseName + '] does not exist. Creating it now.';
END;

DECLARE @CreateSql NVARCHAR(MAX) = N'CREATE DATABASE [' + REPLACE(@DatabaseName, ']', ']]') + N'];';
EXEC (@CreateSql);

DECLARE @UseSql NVARCHAR(MAX) = N'USE [' + REPLACE(@DatabaseName, ']', ']]') + N']; SELECT DB_NAME() AS active_database;';
EXEC (@UseSql);
