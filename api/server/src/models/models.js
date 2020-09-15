'use strict';
module.exports = (sequelize, DataTypes) => {
  const models = sequelize.define('models', {
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
      allowNull: true,
      unique: false
    },
    location: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false,
    },
    superLocation: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false,
    }
  }, {});
  models.associate = function(models) {
    // associations can be defined here
  };
  return models;
};