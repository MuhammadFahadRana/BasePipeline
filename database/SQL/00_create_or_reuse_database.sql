SET NOCOUNT ON;

DECLARE @DatabaseName SYSNAME = N'VideoSemanticDB';

IF DB_ID(@DatabaseName) IS NULL
BEGIN
    PRINT 'Creating database [' + @DatabaseName + ']...';
    DECLARE @CreateSql NVARCHAR(MAX) = N'CREATE DATABASE [' + REPLACE(@DatabaseName, ']', ']]') + N'];';
    EXEC (@CreateSql);
END
ELSE
BEGIN
    PRINT 'Database [' + @DatabaseName + '] already exists. Reusing it.';
END;

DECLARE @UseSql NVARCHAR(MAX) = N'USE [' + REPLACE(@DatabaseName, ']', ']]') + N']; SELECT DB_NAME() AS active_database;';
EXEC (@UseSql);
