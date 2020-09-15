'use strict';
module.exports = (sequelize, DataTypes) => {
  const folders_mains = sequelize.define('folders_mains', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV1,
    },
    projectId: {
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
      unique: false,
      defaultValue: "user",
    },
  }, {});
  folders_mains.associate = function(models) {
    folders_mains.hasMany(models.files_mains, {foreignKey: 'id'})
    // associations can be defined here
  };
  return folders_mains;
};