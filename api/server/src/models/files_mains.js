'use strict';
module.exports = (sequelize, DataTypes) => {
  const files_mains = sequelize.define('files_mains', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV1,
    },
    folderId: {
      type: DataTypes.UUID,
      allowNull: false,
      unique: false,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
    },
    type: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false
    },
    location: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false
    }
  }, {});
  files_mains.associate = function(models) {
    files_mains.belongsTo(models.folders_mains, { foreignKey: 'folderId' });
    // files_mains.hasMany(models.processes, { foreignKey: 'result'});
    // associations can be defined here
  };
  return files_mains;
};